import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ClimateAI", page_icon="🌍", layout="wide")

st.title("🌍 ClimateAI")
st.header("Climate & Environmental Data Analyzer")
st.markdown("*Transform your environmental data into actionable insights with AI-powered analytics*")

# Sidebar
st.sidebar.image("fog.jpg", use_column_width=True)
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Analysis options
if uploaded_file:
    st.sidebar.header("🎯 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📊 Basic Overview", "📈 Time Series Analysis", "🤖 AI Predictions", "🔗 Correlation Analysis", "📋 Data Quality Report", "🌡️ Climate Insights"]
    )

def detect_delimiter(file_content):
    """Smart delimiter detection using multiple methods"""
    import csv
    
    lines = file_content.split('\n')[:10]  # Use first 10 lines for detection
    non_empty_lines = [line for line in lines if line.strip()]
    
    if not non_empty_lines:
        return ','
    
    # Method 1: Use csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff('\n'.join(non_empty_lines[:3])).delimiter
        return delimiter
    except:
        pass
    
    # Method 2: Count delimiters and check consistency
    delimiters = [',', ';', '\t', '|', ':', ' ']
    delimiter_scores = {}
    
    for delim in delimiters:
        counts = [line.count(delim) for line in non_empty_lines]
        if counts and max(counts) > 0:
            # Score based on: frequency, consistency across lines, and reasonableness
            avg_count = sum(counts) / len(counts)
            consistency = 1 - (max(counts) - min(counts)) / (max(counts) + 1)
            reasonableness = min(avg_count / 20, 1)  # Prefer 1-20 fields
            delimiter_scores[delim] = avg_count * consistency * reasonableness
    
    if delimiter_scores:
        best_delim = max(delimiter_scores, key=delimiter_scores.get)
        return best_delim
    
    return ','  # Default fallback

if uploaded_file:
    # Read raw content for analysis
    uploaded_file.seek(0)
    raw_content = uploaded_file.read().decode('utf-8', errors='ignore')
    first_few_lines = raw_content.split('\n')[:5]
    
    # Auto-detect delimiter
    detected_delimiter = detect_delimiter(raw_content)
    
    st.subheader("🔍 File Analysis")
    with st.expander("Show raw file preview"):
        st.text("First few lines of your file:")
        for i, line in enumerate(first_few_lines, 1):
            st.text(f"Line {i}: {repr(line)}")
    
    # Show delimiter detection results
    delimiter_names = {',': 'Comma', ';': 'Semicolon', '\t': 'Tab', '|': 'Pipe', ':': 'Colon', ' ': 'Space'}
    detected_name = delimiter_names.get(detected_delimiter, f"'{detected_delimiter}'")
    
    st.success(f"🎯 Auto-detected delimiter: {detected_name} ('{detected_delimiter}')")
    
    # Reset file pointer for pandas
    uploaded_file.seek(0)
    
    try:
        # Try reading with auto-detected delimiter
        df = pd.read_csv(
            uploaded_file,
            sep=detected_delimiter,
            skipinitialspace=True,
            skip_blank_lines=True,
            on_bad_lines='skip',
            encoding='utf-8',
            engine='python'
        )
        
        # Quick validation - if we get only 1 column, try other delimiters
        if len(df.columns) == 1 and detected_delimiter != '\t':
            st.warning("⚠️ Only 1 column detected, trying alternative delimiters...")
            
            # Try common alternatives
            alt_delimiters = [';', '\t', '|', ':']
            for alt_delim in alt_delimiters:
                if alt_delim != detected_delimiter:
                    try:
                        uploaded_file.seek(0)
                        alt_df = pd.read_csv(
                            uploaded_file,
                            sep=alt_delim,
                            skipinitialspace=True,
                            skip_blank_lines=True,
                            on_bad_lines='skip',
                            encoding='utf-8',
                            engine='python'
                        )
                        if len(alt_df.columns) > 1:
                            df = alt_df
                            detected_delimiter = alt_delim
                            alt_name = delimiter_names.get(alt_delim, f"'{alt_delim}'")
                            st.success(f"✅ Switched to {alt_name} delimiter - found {len(df.columns)} columns!")
                            break
                    except:
                        continue
        
        st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Manual override option (collapsed by default)
        with st.expander("🔧 Manual delimiter override (if needed)"):
            delimiter_options = {',': 'Comma (,)', ';': 'Semicolon (;)', '\t': 'Tab', '|': 'Pipe (|)', ':': 'Colon (:)', ' ': 'Space ( )'}
            manual_delimiter = st.selectbox(
                "Choose different delimiter:",
                options=list(delimiter_options.keys()),
                format_func=lambda x: delimiter_options[x],
                index=list(delimiter_options.keys()).index(detected_delimiter) if detected_delimiter in delimiter_options else 0
            )
            
            if st.button("Apply Manual Delimiter") and manual_delimiter != detected_delimiter:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        sep=manual_delimiter,
                        skipinitialspace=True,
                        skip_blank_lines=True,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        engine='python'
                    )
                    st.success(f"✅ Applied {delimiter_options[manual_delimiter]} - {len(df.columns)} columns detected")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed with manual delimiter: {str(e)}")
        
    except UnicodeDecodeError:
        # Try different encoding if UTF-8 fails
        try:
            df = pd.read_csv(
                uploaded_file,
                sep=detected_delimiter,
                skipinitialspace=True,
                skip_blank_lines=True,
                on_bad_lines='skip',
                encoding='latin-1',
                engine='python'
            )
            st.warning("⚠️ File loaded with latin-1 encoding")
        except Exception as e:
            st.error(f"❌ Encoding error: {str(e)}")
            st.stop()
            
    except pd.errors.ParserError as e:
        st.error(f"❌ CSV parsing error: {str(e)}")
        st.info("💡 Try these solutions:")
        st.markdown("""
        - Check if your CSV has consistent delimiters (commas)
        - Ensure no extra commas in data fields
        - Verify all rows have the same number of columns
        - Remove any empty rows at the beginning or end
        """)
        
        # Offer alternative parsing method
        if st.button("🔧 Try Alternative Parsing"):
            try:
                # More aggressive parsing - treat as tab-separated or detect delimiter
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,  # Let pandas detect delimiter
                    engine='python',
                    skipinitialspace=True,
                    skip_blank_lines=True,
                    on_bad_lines='skip'
                )
                st.success("✅ Successfully loaded with automatic delimiter detection")
            except Exception as alt_e:
                st.error(f"❌ Alternative parsing also failed: {str(alt_e)}")
                st.stop()
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.stop()
    
    # Display data info
    if 'df' in locals():
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Display column names and sample data
        st.subheader("📋 Column Structure")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns],
            'Non-Null Count': [df[col].count() for col in df.columns]
        })
        st.dataframe(col_info)
        
        st.subheader("📊 Data Preview")
        # Show more context about the data structure
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        with col2:
            st.write("**Data shape and info:**")
            st.write(f"Rows: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Column names: {list(df.columns)}")
        
        # Check for common data issues
        st.subheader("⚠️ Data Quality Check")
        issues = []
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            issues.append(f"Found {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        
        # Check for columns with mostly NaN values
        mostly_null_cols = []
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > 80:
                mostly_null_cols.append(f"{col} ({null_pct:.1f}% null)")
        
        if mostly_null_cols:
            issues.append(f"Columns with >80% missing data: {', '.join(mostly_null_cols)}")
        
        # Check for single-column data (might need different delimiter)
        if len(df.columns) == 1:
            issues.append("Only 1 column detected - delimiter might be incorrect")
        
        if issues:
            st.warning("Issues detected:")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ Data looks good!")
            
        # Option to clean data
        if st.button("🧹 Clean Data"):
            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            # Drop columns that are >90% null
            df = df.dropna(axis=1, thresh=int(0.1 * len(df)))
            # Drop rows that are completely empty
            df = df.dropna(how='all')
            st.success(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns remaining")
            st.rerun()
        
        st.subheader("📈 Column Information")
        st.write("**Data types:**")
        st.write(df.dtypes)
        
        # Only proceed with analysis if data loaded successfully
        if len(df) > 0:
            
            # ANALYSIS MODULES
            if analysis_type == "📊 Basic Overview":
                st.header("📊 Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                with col4:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Quick stats
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.subheader("📈 Quick Statistics")
                    st.dataframe(df[numeric_cols].describe())
                    
                    # Distribution plots
                    if len(numeric_cols) <= 6:  # Don't overwhelm with too many plots
                        st.subheader("📊 Data Distributions")
                        fig = make_subplots(
                            rows=2, cols=3,
                            subplot_titles=numeric_cols[:6]
                        )
                        
                        for i, col in enumerate(numeric_cols[:6]):
                            row = (i // 3) + 1
                            col_pos = (i % 3) + 1
                            fig.add_trace(
                                go.Histogram(x=df[col], name=col, showlegend=False),
                                row=row, col=col_pos
                            )
                        
                        fig.update_layout(height=400, title_text="Data Distributions")
                        st.plotly_chart(fig, use_column_width=True)

            elif analysis_type == "📈 Time Series Analysis":
                st.header("📈 Time Series Analysis")
                
                # Try to detect date columns
                date_cols = []
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        date_cols.append(col)
                    except:
                        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                            try:
                                pd.to_datetime(df[col], infer_datetime_format=True, errors='raise')
                                date_cols.append(col)
                            except:
                                continue
                
                if not date_cols:
                    st.warning("⚠️ No date columns detected. Creating index-based time series.")
                    df['Index_Time'] = range(len(df))
                    date_col = 'Index_Time'
                else:
                    date_col = st.selectbox("Select date column:", date_cols)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_metrics = st.multiselect(
                        "Select metrics to analyze:", 
                        numeric_cols,
                        default=numeric_cols[:3]
                    )
                    
                    if selected_metrics:
                        # Create time series plot
                        fig = go.Figure()
                        
                        for metric in selected_metrics:
                            fig.add_trace(go.Scatter(
                                x=df[date_col],
                                y=df[metric],
                                mode='lines+markers',
                                name=metric,
                                line=dict(width=2)
                            ))
                        
                        fig.update_layout(
                            title="Time Series Trends",
                            xaxis_title=date_col,
                            yaxis_title="Values",
                            hovermode='x unified',
                            height=500
                        )
                        st.plotly_chart(fig, use_column_width=True)
                        
                        # Trend analysis
                        st.subheader("📊 Trend Analysis")
                        trend_col1, trend_col2 = st.columns(2)
                        
                        with trend_col1:
                            st.write("**Growth Rates:**")
                            for metric in selected_metrics:
                                if len(df[metric].dropna()) > 1:
                                    first_val = df[metric].dropna().iloc[0]
                                    last_val = df[metric].dropna().iloc[-1]
                                    growth = ((last_val - first_val) / abs(first_val)) * 100
                                    st.write(f"{metric}: {growth:.1f}%")
                        
                        with trend_col2:
                            st.write("**Volatility (Std Dev):**")
                            for metric in selected_metrics:
                                volatility = df[metric].std()
                                st.write(f"{metric}: {volatility:.2f}")

            elif analysis_type == "🤖 AI Predictions":
                st.header("🤖 AI-Powered Predictions")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    target_col = st.selectbox("Select target variable to predict:", numeric_cols)
                    feature_cols = st.multiselect(
                        "Select feature variables:", 
                        [col for col in numeric_cols if col != target_col],
                        default=[col for col in numeric_cols if col != target_col][:5]
                    )
                    
                    if feature_cols and len(df) >= 10:
                        # Prepare data
                        X = df[feature_cols].fillna(df[feature_cols].mean())
                        y = df[target_col].fillna(df[target_col].mean())
                        
                        if len(X) >= 10:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Model selection
                            model_type = st.radio("Choose AI Model:", 
                                                 ["🌳 Random Forest", "📏 Linear Regression"])
                            
                            if model_type == "🌳 Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            else:
                                model = LinearRegression()
                            
                            # Train model
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Model performance
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Absolute Error", f"{mae:.2f}")
                            with col2:
                                st.metric("R² Score", f"{r2:.3f}")
                            
                            # Prediction vs Actual plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=y_test, y=y_pred,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='blue', size=8, opacity=0.6)
                            ))
                            
                            # Perfect prediction line
                            min_val, max_val = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="Actual vs Predicted Values",
                                xaxis_title=f"Actual {target_col}",
                                yaxis_title=f"Predicted {target_col}",
                                height=400
                            )
                            st.plotly_chart(fig, use_column_width=True)
                            
                            # Feature importance (for Random Forest)
                            if model_type == "🌳 Random Forest":
                                st.subheader("🎯 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(importance_df, x='Importance', y='Feature', 
                                           orientation='h', title="Feature Importance Ranking")
                                st.plotly_chart(fig, use_column_width=True)
                            
                            # Future predictions
                            st.subheader("🔮 Make New Predictions")
                            st.write("Enter values for prediction:")
                            
                            prediction_inputs = {}
                            pred_cols = st.columns(min(len(feature_cols), 3))
                            
                            for i, feature in enumerate(feature_cols):
                                col_idx = i % 3
                                with pred_cols[col_idx]:
                                    mean_val = float(df[feature].mean())
                                    prediction_inputs[feature] = st.number_input(
                                        f"{feature}:", 
                                        value=mean_val,
                                        key=f"pred_{feature}"
                                    )
                            
                            if st.button("🎯 Make Prediction"):
                                input_data = np.array([list(prediction_inputs.values())]).reshape(1, -1)
                                prediction = model.predict(input_data)[0]
                                st.success(f"🎯 Predicted {target_col}: **{prediction:.2f}**")
                        
                        else:
                            st.error("Need at least 10 rows for AI predictions")
                    else:
                        st.warning("Please select feature variables for prediction")
                else:
                    st.warning("Need at least 2 numeric columns for AI predictions")

            elif analysis_type == "🔗 Correlation Analysis":
                st.header("🔗 Correlation Analysis")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    # Correlation matrix
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Interactive heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        title="Correlation Heatmap"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_column_width=True)
                    
                    # Strong correlations
                    st.subheader("💪 Strongest Correlations")
                    correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_val):
                                correlations.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Correlation': corr_val,
                                    'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                                })
                    
                    corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(corr_df.head(10))
                    
                    # Scatter plot for selected correlation
                    st.subheader("🔍 Detailed Correlation View")
                    if len(corr_df) > 0:
                        selected_pair = st.selectbox(
                            "Select variable pair:", 
                            [f"{row['Variable 1']} vs {row['Variable 2']}" for _, row in corr_df.head(10).iterrows()]
                        )
                        
                        var1, var2 = selected_pair.split(' vs ')
                        
                        fig = px.scatter(df, x=var1, y=var2, 
                                       trendline="ols",
                                       title=f"Relationship: {var1} vs {var2}")
                        st.plotly_chart(fig, use_column_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")

            elif analysis_type == "📋 Data Quality Report":
                st.header("📋 Data Quality Assessment")
                
                # Missing data analysis
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df)) * 100
                
                quality_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': missing_data,
                    'Missing %': missing_pct,
                    'Data Type': df.dtypes,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                }).sort_values('Missing %', ascending=False)
                
                st.subheader("🔍 Missing Data Analysis")
                
                # Color-code the quality report
                def color_quality(val):
                    if val > 50:
                        return 'background-color: #ffcccc'  # Red for high missing
                    elif val > 20:
                        return 'background-color: #ffffcc'  # Yellow for moderate missing
                    else:
                        return 'background-color: #ccffcc'  # Green for low missing
                
                styled_df = quality_df.style.applymap(color_quality, subset=['Missing %'])
                st.dataframe(styled_df)
                
                # Data quality score
                avg_missing = missing_pct.mean()
                if avg_missing < 5:
                    quality_score = "Excellent 🟢"
                elif avg_missing < 15:
                    quality_score = "Good 🟡"
                elif avg_missing < 30:
                    quality_score = "Fair 🟠"
                else:
                    quality_score = "Poor 🔴"
                
                st.metric("Overall Data Quality", quality_score)
                
                # Recommendations
                st.subheader("💡 Data Quality Recommendations")
                recommendations = []
                
                high_missing = quality_df[quality_df['Missing %'] > 20]
                if len(high_missing) > 0:
                    recommendations.append(f"🔴 Consider removing columns with >20% missing data: {', '.join(high_missing['Column'].tolist())}")
                
                duplicate_rows = df.duplicated().sum()
                if duplicate_rows > 0:
                    recommendations.append(f"🟡 Found {duplicate_rows} duplicate rows - consider removing them")
                
                # Check for potential outliers in numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outlier_cols = []
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                    if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                        outlier_cols.append(col)
                
                if outlier_cols:
                    recommendations.append(f"🟡 Check for outliers in: {', '.join(outlier_cols)}")
                
                if not recommendations:
                    st.success("✅ Your data quality looks good!")
                else:
                    for rec in recommendations:
                        st.write(rec)

            elif analysis_type == "🌡️ Climate Insights":
                st.header("🌡️ Climate & Environmental Insights")
                
                # Try to identify climate-related columns
                climate_keywords = {
                    'temperature': ['temp', 'temperature', 'celsius', 'fahrenheit'],
                    'precipitation': ['rain', 'precipitation', 'rainfall', 'precip'],
                    'humidity': ['humidity', 'humid', 'moisture'],
                    'pressure': ['pressure', 'hpa', 'mbar'],
                    'wind': ['wind', 'breeze', 'gust'],
                    'air_quality': ['pm2.5', 'pm10', 'aqi', 'co2', 'no2', 'so2', 'ozone'],
                    'solar': ['solar', 'radiation', 'uv', 'sunshine']
                }
                
                detected_climate_cols = {}
                for category, keywords in climate_keywords.items():
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in keywords):
                            if category not in detected_climate_cols:
                                detected_climate_cols[category] = []
                            detected_climate_cols[category].append(col)
                
                if detected_climate_cols:
                    st.success(f"🌍 Detected climate data categories: {', '.join(detected_climate_cols.keys())}")
                    
                    # Climate summary dashboard
                    if 'temperature' in detected_climate_cols:
                        temp_cols = detected_climate_cols['temperature']
                        for temp_col in temp_cols[:1]:  # Show first temperature column
                            temp_data = df[temp_col].dropna()
                            if len(temp_data) > 0:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Avg Temperature", f"{temp_data.mean():.1f}°")
                                with col2:
                                    st.metric("Max Temperature", f"{temp_data.max():.1f}°")
                                with col3:
                                    st.metric("Min Temperature", f"{temp_data.min():.1f}°")
                                with col4:
                                    st.metric("Temperature Range", f"{temp_data.max() - temp_data.min():.1f}°")
                    
                    # Environmental risk assessment
                    st.subheader("⚠️ Environmental Risk Assessment")
                    risks = []
                    
                    # Check for extreme temperatures
                    if 'temperature' in detected_climate_cols:
                        temp_col = detected_climate_cols['temperature'][0]
                        temp_data = df[temp_col].dropna()
                        if len(temp_data) > 0:
                            extreme_hot = (temp_data > temp_data.quantile(0.95)).sum()
                            extreme_cold = (temp_data < temp_data.quantile(0.05)).sum()
                            if extreme_hot > len(temp_data) * 0.1:
                                risks.append(f"🔥 High frequency of extreme heat events ({extreme_hot} occurrences)")
                            if extreme_cold > len(temp_data) * 0.1:
                                risks.append(f"🧊 High frequency of extreme cold events ({extreme_cold} occurrences)")
                    
                    # Check air quality
                    if 'air_quality' in detected_climate_cols:
                        aq_col = detected_climate_cols['air_quality'][0]
                        aq_data = df[aq_col].dropna()
                        if len(aq_data) > 0:
                            high_pollution = (aq_data > aq_data.quantile(0.8)).sum()
                            if high_pollution > len(aq_data) * 0.2:
                                risks.append(f"💨 Frequent air quality issues ({high_pollution} high pollution events)")
                    
                    if risks:
                        for risk in risks:
                            st.warning(risk)
                    else:
                        st.success("✅ No major environmental risks detected in the data")
                    
                    # Climate trends visualization
                    st.subheader("📈 Climate Trends")
                    selected_climate_vars = st.multiselect(
                        "Select variables to visualize:",
                        [col for cols in detected_climate_cols.values() for col in cols],
                        default=[col for cols in detected_climate_cols.values() for col in cols][:3]
                    )
                    
                    if selected_climate_vars:
                        fig = go.Figure()
                        for var in selected_climate_vars:
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[var],
                                mode='lines',
                                name=var,
                                line=dict(width=2)
                            ))
                        
                        fig.update_layout(
                            title="Climate Variables Over Time",
                            xaxis_title="Time Index",
                            yaxis_title="Values",
                            height=500,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_column_width=True)
                
                else:
                    st.info("🌍 No specific climate variables detected. Showing general environmental analysis.")
                    
                    # General environmental analysis for any dataset
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        st.subheader("📊 Environmental Data Patterns")
                        
                        # Show variability analysis
                        variability_data = []
                        for col in numeric_cols[:5]:  # Top 5 numeric columns
                            cv = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else 0
                            variability_data.append({'Variable': col, 'Coefficient of Variation': cv})
                        
                        var_df = pd.DataFrame(variability_data).sort_values('Coefficient of Variation', ascending=False)
                        
                        fig = px.bar(var_df, x='Variable', y='Coefficient of Variation',
                                   title="Data Variability Analysis (Higher = More Variable)")
                        st.plotly_chart(fig, use_column_width=True)

else:
    # Enhanced welcome screen
    st.info("👆 **Upload your environmental data to unlock powerful AI insights!**")
    
    # Create sample data showcase
    st.header("✨ What ClimateAI Can Do For You")
    
    # Feature showcase
    features = [
        {"title": "📊 Smart Data Analysis", "desc": "Automatic delimiter detection, data quality assessment, and intelligent preprocessing"},
        {"title": "🤖 AI-Powered Predictions", "desc": "Random Forest and Linear Regression models for environmental forecasting"},
        {"title": "📈 Time Series Insights", "desc": "Trend analysis, seasonality detection, and growth rate calculations"},
        {"title": "🔗 Correlation Discovery", "desc": "Interactive heatmaps and relationship analysis between variables"},
        {"title": "🌡️ Climate Intelligence", "desc": "Specialized analysis for temperature, air quality, precipitation, and more"},
        {"title": "📋 Quality Reports", "desc": "Comprehensive data health checks with actionable recommendations"}
    ]
    
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        with col1:
            if i < len(features):
                st.subheader(features[i]["title"])
                st.write(features[i]["desc"])
        with col2:
            if i + 1 < len(features):
                st.subheader(features[i+1]["title"])
                st.write(features[i+1]["desc"])
    
    # Sample data generator
    st.header("🎯 Try It Now - Generate Sample Climate Data")
    if st.button("🌡️ Generate Sample Dataset", type="primary"):
        # Create sample climate data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Temperature_C': 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365),
            'Humidity_%': 60 + 20 * np.sin(2 * np.pi * np.arange(365) / 365 + np.pi/4) + np.random.normal(0, 5, 365),
            'Air_Quality_AQI': 50 + 30 * np.random.random(365) + 10 * np.sin(2 * np.pi * np.arange(365) / 365),
            'Wind_Speed_kmh': 15 + 5 * np.random.random(365),
            'Precipitation_mm': np.maximum(0, np.random.exponential(2, 365))
        })
        
        # Apply sample_data to the session
        st.session_state['sample_data'] = sample_data
        
        st.success("✅ Sample dataset generated! Here's a preview:")
        st.dataframe(sample_data.head())
        
        # Quick analysis of sample data
        st.subheader("📊 Quick Sample Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Temperature", f"{sample_data['Temperature_C'].mean():.1f}°C")
        with col2:
            st.metric("Max AQI", f"{sample_data['Air_Quality_AQI'].max():.0f}")
        with col3:
            st.metric("Total Precipitation", f"{sample_data['Precipitation_mm'].sum():.0f}mm")
        
        # Sample visualization
        fig = px.line(sample_data, x='Date', y='Temperature_C', 
                     title='Sample Temperature Data - Seasonal Pattern')
        st.plotly_chart(fig, use_column_width=True)
        
        # Download sample
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv_sample,
            file_name="sample_climate_data.csv",
            mime="text/csv"
        )
    
    # Tips section
    st.header("💡 Data Upload Tips")
    tips = [
        "**Supported formats**: CSV files with any delimiter (comma, semicolon, tab, etc.)",
        "**Climate data**: Temperature, humidity, air quality, precipitation work best",
        "**Time series**: Include date columns for trend analysis",
        "**Size**: Files up to 200MB supported",
        "**Quality**: Missing values are handled automatically"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    # Add some visual appeal
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin: 20px 0;'>
        <h3 style='color: #2E8B57;'>🌍 Transform Your Environmental Data Into Actionable Insights</h3>
        <p style='font-size: 16px; color: #666;'>Upload your CSV file above to get started with advanced climate analytics</p>
    </div>
    """, unsafe_allow_html=True)