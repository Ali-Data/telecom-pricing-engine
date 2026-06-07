import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# --- تنظیمات اولیه صفحه ---
st.set_page_config(page_title="AI Pricing Optimizer", page_icon="🚀", layout="wide")
st.title("🚀 سیستم هوشمند بهینه‌سازی تخفیف و مدیریت ریزش")
st.markdown("این داشبورد فایل مشتریان در معرض خطر (نزدیک به انقضا) را دریافت کرده و با استفاده از هوش مصنوعی، بهترین استراتژی حفظ مشتری را ارائه می‌دهد.")

# --- بارگذاری هسته هوش مصنوعی ---
@st.cache_resource
def load_ai_brain():
    # فرض بر این است که داشبورد از پوشه اصلی پروژه اجرا می‌شود
    model_path = "ml_pipeline/calibrated_xgb_model.pkl"
    features_path = "ml_pipeline/feature_names.pkl"
    
    if not os.path.exists(model_path):
        # بررسی مسیر جایگزین اگر مستقیماً از داخل پوشه app اجرا شد
        model_path = "../ml_pipeline/calibrated_xgb_model.pkl"
        features_path = "../ml_pipeline/feature_names.pkl"
        
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: فایل‌های مدل یافت نشدند.")
        return None, None

model, feature_names = load_ai_brain()

# --- سایدبار و دریافت فایل ---
st.sidebar.header("📥 ورود اطلاعات مشتریان")
st.sidebar.info("فایل خروجی دیتابیس (مثلاً ۵ روز مانده به انقضا) را در این بخش آپلود کنید.")
uploaded_file = st.sidebar.file_uploader("آپلود فایل CSV", type=["csv"])

if uploaded_file is not None and model is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write(f"✅ تعداد **{len(df_input)}** مشتری با موفقیت بارگذاری شد.")
    
    if st.button("🧠 شروع پردازش و محاسبه بهترین تخفیف‌ها", type="primary"):
        with st.spinner('هوش مصنوعی در حال شبیه‌سازی کشش قیمتی و امید ریاضی...'):
            
            # --- منطق شبیه‌ساز (شبیه‌سازی جهان‌های موازی) ---
            df_sim = df_input.copy()
            df_sim['temp_user_id'] = range(len(df_sim))
            
            # محاسبه ARPU قبلی برای حفظ منطق روندها
            if 'arpu_trend' in df_sim.columns and 'arpu' in df_sim.columns:
                df_sim['prev_arpu_actual'] = df_sim['arpu'] - df_sim['arpu_trend']
            else:
                st.error("ستون‌های arpu و arpu_trend در فایل شما وجود ندارد!")
                st.stop()
                
            # سناریوهای تخفیف (۰ تا ۳۰ درصد)
            discount_scenarios = np.linspace(0.0, 0.30, 7)
            df_expanded = df_sim.loc[df_sim.index.repeat(len(discount_scenarios))].reset_index(drop=True)
            df_expanded['simulated_discount'] = np.tile(discount_scenarios, len(df_sim))
            
            # آپدیت فیچرها بر اساس تخفیف اعمال شده
            df_expanded['hidden_discount_percentage'] = df_expanded['simulated_discount']
            df_expanded['offered_price'] = df_expanded['catalog_price'] * (1 - df_expanded['simulated_discount'])
            df_expanded['hidden_discount_amount'] = df_expanded['catalog_price'] - df_expanded['offered_price']
            
            df_expanded['arpu'] = np.where(df_expanded['duration'] > 0, df_expanded['offered_price'] / df_expanded['duration'], 0)
            df_expanded['arpu_trend'] = df_expanded['arpu'] - df_expanded['prev_arpu_actual']
            
            # آماده‌سازی دیتا برای ورود به ماشین
            try:
                X_predict = df_expanded[feature_names]
            except KeyError as e:
                st.error(f"فایل آپلودی تمام ستون‌های مورد نیاز مدل را ندارد. ستون‌های گمشده: {e}")
                st.stop()
                
            # پیش‌بینی احتمالات
            probs = model.predict_proba(X_predict)
            df_expanded['Prob_Accept'] = probs[:, 0]
            df_expanded['Prob_Downgrade'] = probs[:, 1]
            df_expanded['Prob_Churn'] = probs[:, 2]
            
            # محاسبه سود مورد انتظار (Expected Value)
            DOWNGRADE_REVENUE_RATIO = 0.70
            df_expanded['Expected_Value'] = (df_expanded['Prob_Accept'] * df_expanded['offered_price']) + \
                                            (df_expanded['Prob_Downgrade'] * (df_expanded['offered_price'] * DOWNGRADE_REVENUE_RATIO))
                                            
            # استخراج بهترین سناریو برای هر فرد
            optimal_scenarios = df_expanded.loc[df_expanded.groupby('temp_user_id')['Expected_Value'].idxmax()].copy()
            
            # --- نمایش نتایج ---
            st.success("🎉 پردازش به پایان رسید!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 توزیع تخفیف‌ها")
                discount_distribution = optimal_scenarios['simulated_discount'].value_counts(normalize=True).sort_index() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar([f"{int(d*100)}%" for d in discount_distribution.index], discount_distribution.values, color='cornflowerblue', edgecolor='black')
                ax.set_ylabel('درصد کاربران')
                st.pyplot(fig)
                
            with col2:
                st.subheader("📋 لیست نهایی عملیاتی برای مارکتینگ")
                
                # استخراج ستون‌های حیاتی که مدیر مارکتینگ نیاز دارد ببیند
                cols_to_show = ['username', 'arpu']
                if 'rolling_3_disruptioncount' in optimal_scenarios.columns:
                    cols_to_show.append('rolling_3_disruptioncount')
                    
                cols_to_show.extend(['catalog_price', 'simulated_discount', 'offered_price'])
                final_output = optimal_scenarios[cols_to_show].copy()
                
                # زیباسازی داده‌ها برای نمایش
                final_output['simulated_discount'] = (final_output['simulated_discount'] * 100).astype(int).astype(str) + "%"
                final_output['arpu'] = final_output['arpu'].astype(int)
                final_output['catalog_price'] = final_output['catalog_price'].astype(int)
                final_output['offered_price'] = final_output['offered_price'].astype(int)
                
                # تغییر نام ستون‌ها به فارسی برای تیم فروش
                rename_dict = {
                    'username': 'شناسه کاربر',
                    'arpu': 'ارزش ماهانه (ARPU)',
                    'rolling_3_disruptioncount': 'میانگین قطعی (۳ خرید اخیر)',
                    'catalog_price': 'قیمت بدون تخفیف',
                    'simulated_discount': 'تخفیف هوش مصنوعی',
                    'offered_price': 'مبلغ نهایی پرداختی'
                }
                final_output.rename(columns=rename_dict, inplace=True)
                
                st.dataframe(final_output.head(15), use_container_width=True)
                
            # ایجاد امکان دانلود خروجی
            st.markdown("---")
            csv_data = final_output.to_csv(index=False).encode('utf-8-sig') # utf-8-sig برای پشتیبانی از فارسی در اکسل
            st.download_button(
                label="📥 دانلود کامل لیست برای ارسال پیامک (CSV)",
                data=csv_data,
                file_name='marketing_ai_discounts.csv',
                mime='text/csv',
                type="primary"
            )
else:
    st.info("👈 برای شروع، لطفاً فایل CSV مشتریان را از منوی سمت چپ آپلود کنید.")