import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import skew
import warnings
from google import genai
from google.genai import types
from google.genai.errors import APIError
import json

# Suppress warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="快速客戶分群 (Fast Clustering)", layout="wide", page_icon="👥")

# ============================================================
# Core Logic & Caching
# ============================================================

@st.cache_data
def load_data(file):
    """Load data with caching"""
    return pd.read_csv(file)

def smart_preprocessing(data, features):
    """
    Intelligently preprocess data: Log transform if skewed, then Scale.
    """
    df_clean = data[features].dropna()
    
    transform_info = {
        'log_features': [],
        'scaler': None
    }
    
    processed_data = df_clean.copy()
    
    for feat in features:
        if pd.api.types.is_numeric_dtype(processed_data[feat]) and processed_data[feat].min() >= 0:
            if skew(processed_data[feat]) > 1:
                processed_data[feat] = np.log1p(processed_data[feat])
                transform_info['log_features'].append(feat)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    transform_info['scaler'] = scaler
    
    return scaled_data, transform_info, df_clean

def calculate_silhouette_sample(X, labels, sample_size=5000):
    """
    Calculate Silhouette Score on a sample to avoid memory crashes, excluding noise labels (-1).
    """
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    if len(unique_labels) < 2:
        return -1 
    
    X_clust = X[mask]
    labels_clust = labels[mask]
    
    if len(X_clust) > sample_size:
        indices = np.random.choice(len(X_clust), sample_size, replace=False)
        X_sample = X_clust[indices]
        labels_sample = labels_clust[indices]
        return silhouette_score(X_sample, labels_sample)
    else:
        return silhouette_score(X_clust, labels_clust)

def run_clustering_optimized(X, n_clusters_range):
    """Run clustering algorithms with Memory Safety Checks"""
    results = {}
    rows = X.shape[0]
    
    st.info(f"正在分析 {rows:,} 筆資料...")

    # --- 1. K-Means ---
    st.write(f"⏳ 正在執行 K-Means (範圍: {n_clusters_range})...")
    best_score_km = -1
    best_model_km = None
    
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        score = calculate_silhouette_sample(X, labels)
        
        if score > best_score_km:
            best_score_km = score
            best_model_km = (k, labels, model)
            
    results['K-Means'] = {
        'labels': best_model_km[1],
        'score': best_score_km,
        'params': f"k={best_model_km[0]}",
        'model': best_model_km[2]
    }

    # --- MEMORY SAFETY CHECK ---
    if rows > 15000:
        st.warning(f"⚠️ 資料量過大 ({rows:,} 筆)。已自動跳過 'Agglomerative' 與 'DBSCAN' 以避免記憶體崩潰 (Memory Error)。")
        st.caption("階層式分群與密度分群在超過 1.5 萬筆資料時極耗資源，目前僅執行 K-Means。")
        return results

    # --- 2. Agglomerative ---
    st.write("⏳ 正在執行 Agglomerative Clustering...")
    # ... (Agglomerative clustering logic remains the same) ...
    best_score_agg = -1
    best_res_agg = None
    try:
        for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            score = calculate_silhouette_sample(X, labels)
            
            if score > best_score_agg:
                best_score_agg = score
                best_res_agg = (k, labels)
        if best_res_agg:
            results['Agglomerative'] = {
                'labels': best_res_agg[1],
                'score': best_score_agg,
                'params': f"k={best_res_agg[0]}",
                'model': None
            }
        else:
             results['Agglomerative'] = {'labels': np.full(rows, -1), 'score': -1, 'params': 'Failed', 'model': None}
    except Exception as e:
        st.warning(f"Skipping Agglomerative due to error: {e}")


    # --- 3. DBSCAN ---
    st.write("⏳ 正在執行 DBSCAN...")
    # ... (DBSCAN clustering logic remains the same) ...
    best_score_db = -1
    best_res_db = None
    eps_range = np.arange(0.5, 1.5, 0.5) 
    min_samples_range = [5, 10]
    try:
        for eps in eps_range:
            for ms in min_samples_range:
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                unique_labels = set(labels)
                if -1 in unique_labels: unique_labels.remove(-1)
                
                if 1 < len(unique_labels) < 20:
                    score = calculate_silhouette_sample(X, labels)
                    if score > best_score_db:
                        best_score_db = score
                        best_res_db = (eps, ms, labels)
        
        if best_res_db:
            results['DBSCAN'] = {
                'labels': best_res_db[2],
                'score': best_score_db,
                'params': f"eps={best_res_db[0]:.1f}, min={best_res_db[1]}",
                'model': None
            }
        else:
             results['DBSCAN'] = {'labels': np.full(rows, -1), 'score': -1, 'params': 'Failed', 'model': None}
    except Exception as e:
         st.warning(f"Skipping DBSCAN due to error: {e}")

    return results

# ============================================================
# LLM Function (Modified for Structured JSON Output)
# ============================================================

def generate_cluster_descriptions(df_viz, features, overall_means, api_key=None, threshold=0.15):
    """Generates a structured JSON description for each cluster using Gemini if API key provided."""
    
    grouped_means = df_viz[df_viz['Cluster'] != '-1'].groupby('Cluster')[features].mean()
    
    descriptions = {}
    
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            
            # --- MODIFIED PROMPT & SCHEMA FOR STABILITY ---
            prompt = f"""
            你是一位資深資料科學家，請根據以下客戶分群結果，為每個群組提供**簡潔**的業務解讀。
            
            請**嚴格**以 JSON 格式輸出，格式為: {{ "cluster_ID": {{ "輪廓": "...", "特徵": "...", "策略": "..." }} for each cluster }}
            
            **解讀內容要求 (需使用中文)**:
            1. **輪廓**: 描述該群組的整體客戶性質 (如：高價值客戶、休眠客戶)。
            2. **特徵**: **簡潔**地列出與總體平均相比，偏差超過 {int(threshold*100)}% 的關鍵特徵 (使用「高」或「低」)。例如: 高刷卡金額，低活躍度。
            3. **策略**: 提出一個簡潔有力的行銷或業務策略建議。

            ---
            
            **資料與結果**
            
            特徵列表: {', '.join(features)}
            總體平均 (Overall Means): {overall_means.to_dict()}
            群組平均 (Cluster Means):
            {grouped_means.to_string()}
            """
            
            # Define the nested JSON structure
            cluster_properties = {
                "輪廓": {"type": "string", "description": "該群組的整體客戶性質描述"},
                "特徵": {"type": "string", "description": "簡潔的關鍵特徵列表 (使用高/低)"},
                "策略": {"type": "string", "description": "簡潔的行銷或業務策略建議"},
            }
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            f"cluster_{id}": {"type": "object", "properties": cluster_properties} 
                            for id in grouped_means.index
                        }
                    }
                ),
            )
            
            llm_output = response.text
            # We assume the output is a dictionary of dictionaries
            llm_descriptions = json.loads(llm_output)
            
            # Reformat the result to match the expected format for fallback if needed
            for cluster_id, data in llm_descriptions.items():
                if isinstance(data, dict):
                    # Combine structured fields into a single dict for consistent handling
                    descriptions[str(cluster_id).replace('cluster_', '')] = data
                
            st.success("✅ 已使用 Gemini API 生成結構化業務解讀。")
            
        except APIError as e:
            st.warning(f"Gemini API 呼叫失敗: {e}. 請檢查您的 API Key 或 API 額度。使用預設靜態解讀。")
        except Exception as e:
            st.warning(f"LLM 解讀生成失敗: {e}. 使用預設靜態解讀。")
    
    # Fallback to hardcoded (if API failed or key not provided)
    if not descriptions or not api_key:
        for cluster_id, row in grouped_means.iterrows():
            high_feats = []
            low_feats = []
            
            for feat in features:
                cluster_mean = row[feat]
                overall_mean = overall_means[feat]
                deviation = (cluster_mean - overall_mean) / overall_mean if overall_mean != 0 else 0
                
                if deviation > threshold: 
                    high_feats.append(feat)
                elif deviation < -threshold:
                    low_feats.append(feat)
            
            # Generate structured output even for fallback
            if not high_feats and not low_feats:
                descriptions[str(cluster_id)] = {
                    "輪廓": "平均型客戶",
                    "特徵": "與總體平均無顯著差異",
                    "策略": "標準化行銷活動，維持既有關係。"
                }
            else:
                desc_feats = ""
                if high_feats:
                    desc_feats += f"高{'、'.join(high_feats)}"
                if high_feats and low_feats:
                    desc_feats += "，"
                if low_feats:
                    desc_feats += f"低{'、'.join(low_feats)}"
                
                descriptions[str(cluster_id)] = {
                    "輪廓": "差異化特徵客戶",
                    "特徵": desc_feats,
                    "策略": "針對關鍵特徵制定專屬行銷活動，提升客戶價值。"
                }
    
    return descriptions, grouped_means

# ============================================================
# UI Components (Modified Display Logic)
# ============================================================

def main():
    st.title("👥 快速客戶分群系統 (Fast Clustering)")

    # 1. Sidebar: Upload & Settings
    with st.sidebar:
        st.header("1. 上傳資料")
        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=['csv'])
        st.divider()
        st.header("2. 參數設定")
        min_c = st.number_input("最小群組數 (Min Clusters)", 2, 5, 2)
        max_c = st.number_input("最大群組數 (Max Clusters)", 6, 15, 8)
        
        gemini_api_key = st.text_input("Google Gemini API Key (選填，用於動態 LLM 解讀)", type="password")
        if gemini_api_key:
            st.caption("使用 Gemini API 需有有效 API Key。")

    if uploaded_file:
        df = load_data(uploaded_file)
        
        st.subheader("原始資料預覽")
        st.dataframe(df.head(10))
        
        st.subheader("選擇分群特徵")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("資料中沒有數值欄位，無法進行分群。")
            return

        selected_features = st.multiselect(
            "請選擇正好 2 或 3 個用於分群的欄位 (以便於視覺化):", 
            numeric_cols
        )
        
        if len(selected_features) not in [2, 3]:
            st.warning("⚠️ 請選擇正好 2 或 3 個特徵進行分析。")
        else:
            if st.button("🚀 開始分群運算 (Start Clustering)", type="primary"):
                with st.spinner("正在處理資料與運算模型..."):
                    X_scaled, transform_info, df_used = smart_preprocessing(df, selected_features)
                    overall_means = df_used[selected_features].mean()
                    results = run_clustering_optimized(X_scaled, (min_c, max_c))
                    
                    st.session_state.results = results
                    st.session_state.transform_info = transform_info
                    st.session_state.features = selected_features
                    st.session_state.X_scaled = X_scaled
                    st.session_state.df_used = df_used
                    st.session_state.overall_means = overall_means
                    st.session_state.api_key = gemini_api_key
                    st.session_state.ran = True

            # --- Display Results Section ---
            if st.session_state.get('ran', False):
                st.divider()
                st.header("🎯 分群結果")
                
                results = st.session_state.results
                
                # Compare Models Table (Unchanged)
                comp_data = []
                valid_results = {name: res for name, res in results.items() if res['score'] > -1}
                for name, res in valid_results.items():
                    labels_no_noise = res['labels'][res['labels'] != -1]
                    n_clus = len(set(labels_no_noise)) if len(labels_no_noise) > 0 else 0
                    comp_data.append({'模型': name, '輪廓係數 (Score)': f"{res['score']:.4f}", '分群數量': n_clus, '最佳參數': res['params']})
                
                if not comp_data:
                    st.error("所有分群方法皆失敗，請檢查資料品質或選取的特徵。")
                    return
                
                df_comp = pd.DataFrame(comp_data).sort_values('輪廓係數 (Score)', ascending=False)
                st.table(df_comp)
                
                best_model_name = df_comp['模型'].iloc[0]
                best_res = results[best_model_name]
                st.success(f"🏆 最佳模型: **{best_model_name}** (Score: {best_res['score']:.4f})")
                
                df_viz = st.session_state.df_used.copy()
                df_viz['Cluster'] = best_res['labels'].astype(str)
                
                # --- Cluster Interpretation (Modified Display Logic) ---
                st.subheader("💡 分群業務解讀 (Cluster Interpretation)")
                descriptions, grouped_means = generate_cluster_descriptions(
                    df_viz, 
                    st.session_state.features, 
                    st.session_state.overall_means,
                    api_key=st.session_state.get('api_key')
                )

                # 1. Display Mean Table (Unchanged)
                st.write("**群組平均特徵值比較 (Cluster Mean Features)**")
                display_means = grouped_means.drop(index='-1', errors='ignore').round(1)
                overall_df = pd.DataFrame([st.session_state.overall_means.round(1)], index=['Overall Mean'])
                st.dataframe(pd.concat([display_means, overall_df]))
                
                # 2. Display Descriptions in Simple List Format (New Logic)
                st.markdown("## 📋 簡潔解讀清單")
                st.markdown("---") 

                for cluster_id, structured_desc in descriptions.items():
                    if cluster_id == '-1':
                        continue
                    
                    # 使用 .get() 確保即使 JSON 結構不完整也不會崩潰
                    profile_display = structured_desc.get("輪廓", "客戶輪廓未知")
                    core_features = structured_desc.get("特徵", "與總體平均無顯著差異")
                    strategy = structured_desc.get("策略", "無明確策略建議")
                    
                    st.markdown(f"### 📍 Cluster {cluster_id}")
                    st.markdown(f"""
                    * **客戶輪廓**: {profile_display}
                    * **核心特徵**: {core_features}
                    * **建議策略**: {strategy}
                    """)
                    st.markdown("---")

                # Visualization (Unchanged)
                # ... (Visualization logic remains the same) ...
                col1, col2 = st.columns([3, 1])
                with col1:
                    df_viz_clustered = df_viz[df_viz['Cluster'] != '-1']
                    
                    if len(selected_features) == 2:
                        fig = px.scatter(df_viz_clustered, x=selected_features[0], y=selected_features[1], color='Cluster', title=f"分群視覺化 ({best_model_name}) - 2D Scatter", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(selected_features) == 3:
                        fig = px.scatter_3d(df_viz_clustered, x=selected_features[0], y=selected_features[1], z=selected_features[2], color='Cluster', title=f"分群視覺化 ({best_model_name}) - 3D Scatter", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**群組大小統計 (Count)**")
                    stats = df_viz_clustered['Cluster'].value_counts().reset_index()
                    stats.columns = ['Cluster', 'Count']
                    st.dataframe(stats, use_container_width=True)

                # Prediction Section (Unchanged)
                # ... (Prediction logic remains the same) ...
                st.divider()
                st.header("🔍 單筆預測")
                with st.expander("輸入數值進行預測", expanded=False):
                    features = st.session_state.features
                    transform_info = st.session_state.transform_info
                    
                    inputs = {}
                    cols = st.columns(len(features))
                    for i, feat in enumerate(features):
                        with cols[i]:
                            inputs[feat] = st.number_input(f"{feat}", value=float(st.session_state.df_used[feat].mean()))
                    
                    if st.button("預測所屬群組"):
                        input_df = pd.DataFrame([inputs])
                        for feat in transform_info['log_features']:
                            input_df[feat] = np.log1p(input_df[feat])
                        input_scaled = transform_info['scaler'].transform(input_df)
                        
                        pred_label = -1
                        if best_model_name == 'K-Means' and best_res['model'] is not None:
                            pred_label = best_res['model'].predict(input_scaled)[0]
                        else:
                            df_temp = pd.DataFrame(st.session_state.X_scaled)
                            df_temp['label'] = best_res['labels']
                            active_clusters = df_temp[df_temp['label'] != -1]
                            if not active_clusters.empty:
                                centroids = active_clusters.groupby('label')[list(range(len(features)))].mean().values
                                unique_labels = sorted(active_clusters['label'].unique())
                                dists = np.linalg.norm(centroids - input_scaled, axis=1)
                                pred_label = unique_labels[dists.argmin()]
                        
                        if pred_label != -1:
                            st.success(f"### 該客戶屬於: Cluster {pred_label}")
                            # Show structured interpretation of the predicted cluster
                            if str(pred_label) in descriptions:
                                pred_desc = descriptions[str(pred_label)]
                                st.markdown(f"""
                                **群組 {pred_label} 輪廓**: 
                                * **特徵**: {pred_desc.get('特徵', 'N/A')}
                                * **策略**: {pred_desc.get('策略', 'N/A')}
                                """)
                        else:
                            st.warning("無法將此客戶分類至任何有效群組。")

if __name__ == "__main__":
    if 'ran' not in st.session_state:
        st.session_state.ran = False
        
    main()