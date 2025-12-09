import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import skew
import warnings
import json
from typing import Tuple, Dict, Optional, List
from functools import lru_cache
import requests # NEW: Import for HuggingFace API calls
# from google import genai # REMOVED: Gemini imports
# from google.genai import types # REMOVED
# from google.genai.errors import APIError # REMOVED

# Advanced clustering packages
try:
    from kmodes.kmodes import KModes
    import gower
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="全方位客戶分群 (Advanced Clustering)", 
    layout="wide", 
    page_icon="👥"
)

# ============================================================
# Configuration & Constants
# ============================================================

class Config:
    """Centralized configuration"""
    MAX_GOWER_ROWS = 5000
    SILHOUETTE_SAMPLE_SIZE = 5000
    MIN_SAMPLES_FOR_SILHOUETTE = 50
    CACHE_TTL = 3600
    DEFAULT_RANDOM_STATE = 42
    
    # HuggingFace Configuration (NEW)
    HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"




# ============================================================
# Data Processing & Caching
# ============================================================

@st.cache_data(ttl=Config.CACHE_TTL)
def load_data(file) -> pd.DataFrame:
    """Load and validate CSV data with caching"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("Uploaded file is empty")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        raise

def detect_column_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    """
    Efficiently detect numeric and categorical columns
    Returns: (numeric_cols, categorical_cols)
    """
    df_subset = df[features]
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_subset.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols

def smart_preprocessing_numeric(
    data: pd.DataFrame, 
    features: List[str], 
    log_threshold: float = 1.0
) -> Tuple[np.ndarray, Dict, pd.DataFrame]:
    """
    Optimized preprocessing with better memory management
    - Handles missing values efficiently
    - Applies log transformation only when needed
    - Uses in-place operations where possible
    """
    df_clean = data[features].dropna().copy()
    
    if df_clean.empty:
        raise ValueError("No valid data after removing missing values")
    
    transform_info = {
        'log_features': [],
        'scaler': None,
        'feature_stats': {}
    }
    
    # Process features efficiently
    for feat in features:
        if pd.api.types.is_numeric_dtype(df_clean[feat]):
            col_min = df_clean[feat].min()
            col_skew = skew(df_clean[feat])
            
            transform_info['feature_stats'][feat] = {
                'min': col_min,
                'max': df_clean[feat].max(),
                'skew': col_skew
            }
            
            # Apply log transform only if data is non-negative and highly skewed
            if col_min >= 0 and col_skew > log_threshold:
                df_clean[feat] = np.log1p(df_clean[feat])
                transform_info['log_features'].append(feat)
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    transform_info['scaler'] = scaler
    
    return scaled_data, transform_info, df_clean

def calculate_multiple_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple clustering quality metrics for better evaluation
    """
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    metrics = {
        'silhouette': -1,
        'calinski_harabasz': -1,
        'davies_bouldin': float('inf'),
        'n_clusters': len(unique_labels),
        'noise_ratio': np.sum(labels == -1) / len(labels)
    }
    
    if len(unique_labels) < 2:
        return metrics
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    try:
        # Silhouette Score (higher is better)
        if len(X_clean) <= Config.SILHOUETTE_SAMPLE_SIZE:
            metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
        else:
            # Sample for large datasets
            indices = np.random.choice(len(X_clean), Config.SILHOUETTE_SAMPLE_SIZE, replace=False)
            metrics['silhouette'] = silhouette_score(X_clean[indices], labels_clean[indices])
        
        # Calinski-Harabasz Score (higher is better)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
        
        # Davies-Bouldin Score (lower is better)
        metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
        
    except Exception as e:
        st.warning(f"Error calculating metrics: {e}")
    
    return metrics

def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """
    Calculate composite score combining multiple metrics
    Normalizes and weights different metrics appropriately
    """
    # Normalize silhouette (-1 to 1) -> (0 to 1)
    sil_norm = (metrics['silhouette'] + 1) / 2
    
    # Penalize noise
    noise_penalty = 1 - metrics['noise_ratio']
    
    # Davies-Bouldin: lower is better, invert for composite score
    # Typical range is 0 to infinity, cap at 10 for normalization
    db_norm = 1 / (1 + min(metrics['davies_bouldin'], 10))
    
    # Weighted composite (adjust weights as needed)
    composite = (
        0.5 * sil_norm +
        0.3 * db_norm +
        0.2 * noise_penalty
    )
    
    return composite

# ============================================================
# Clustering Algorithms (Optimized)
# ============================================================

def run_kmeans(X: np.ndarray, n_clusters_range: Tuple[int, int]) -> Dict:
    """Optimized K-Means with better model selection"""
    results = {}
    best_score = -1
    best_model = None
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running K-Means (range: {n_clusters_range})...")
    
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        model = KMeans(
            n_clusters=k, 
            random_state=Config.DEFAULT_RANDOM_STATE,
            n_init=10,  # Multiple initializations for stability
            max_iter=300
        )
        labels = model.fit_predict(X)
        metrics = calculate_multiple_metrics(X, labels)
        composite_score = calculate_composite_score(metrics)
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = (k, labels, model, metrics)
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'K-Means',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[3],
            'params': f"k={best_model[0]}",
            'model': best_model[2]
        }
    else:
        results['Best Model'] = {'score': -1, 'labels': [], 'type': 'Failed'}
    
    return results

def run_kmodes(df: pd.DataFrame, n_clusters_range: Tuple[int, int]) -> Dict:
    """Optimized K-Modes for categorical data"""
    if not ADVANCED_AVAILABLE:
        st.error("K-Modes not available. Install: pip install kmodes")
        return {'Best Model': {'score': -1, 'type': 'K-Modes Unavailable'}}
    
    results = {}
    best_score = -1
    best_model = None
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running K-Modes (range: {n_clusters_range})...")
    
    X_matrix = df.values
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    # Pre-encode for metric calculation
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    X_encoded = df_encoded.values
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        try:
            km = KModes(
                n_clusters=k, 
                init='Huang', 
                n_init=5, 
                verbose=0, 
                random_state=Config.DEFAULT_RANDOM_STATE
            )
            labels = km.fit_predict(X_matrix)
            
            metrics = calculate_multiple_metrics(X_encoded, labels)
            composite_score = calculate_composite_score(metrics)
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = (k, labels, km, metrics)
        except Exception as e:
            st.warning(f"K-Modes failed for k={k}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'K-Modes',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[3],
            'params': f"k={best_model[0]}",
            'model': best_model[2]
        }
    else:
        results['Best Model'] = {'score': -1, 'type': 'Failed'}
    
    return results

def run_gower_hierarchical(df: pd.DataFrame, n_clusters_range: Tuple[int, int]) -> Dict:
    """Optimized Gower + Hierarchical with memory safety"""
    if not ADVANCED_AVAILABLE:
        st.error("Gower clustering not available. Install: pip install gower")
        return {'Best Model': {'score': -1, 'type': 'Gower Unavailable'}}
    
    results = {}
    rows = df.shape[0]
    
    # Memory safety check
    if rows > Config.MAX_GOWER_ROWS:
        st.error(
            f"⚠️ Data exceeds {Config.MAX_GOWER_ROWS} rows. "
            "Gower distance calculation requires too much memory. "
            "Please sample your data or use numeric-only features."
        )
        return {'Best Model': {'score': -1, 'type': 'Error: Data too large'}}
    
    st.write("⏳ Computing Gower Distance Matrix (this may take time)...")
    
    try:
        with st.spinner("Calculating distances..."):
            dist_matrix = gower.gower_matrix(df)
    except Exception as e:
        st.error(f"Gower calculation failed: {e}")
        return {'Best Model': {'score': -1, 'type': 'Error'}}
    
    progress_bar = st.progress(0)
    st.write(f"⏳ Running Hierarchical Clustering (range: {n_clusters_range})...")
    
    best_score = -1
    best_model = None
    total_iterations = n_clusters_range[1] - n_clusters_range[0] + 1
    
    for idx, k in enumerate(range(n_clusters_range[0], n_clusters_range[1] + 1)):
        try:
            model = AgglomerativeClustering(
                n_clusters=k, 
                metric='precomputed', 
                linkage='average'
            )
            labels = model.fit_predict(dist_matrix)
            
            # Calculate metrics with precomputed distance
            mask = labels != -1
            unique_labels = np.unique(labels[mask])
            
            metrics = {
                'silhouette': -1,
                'n_clusters': len(unique_labels),
                'noise_ratio': 0
            }
            
            if len(unique_labels) >= 2:
                dist_clean = dist_matrix[mask][:, mask]
                labels_clean = labels[mask]
                
                if len(dist_clean) <= Config.SILHOUETTE_SAMPLE_SIZE:
                    metrics['silhouette'] = silhouette_score(
                        dist_clean, labels_clean, metric='precomputed'
                    )
                else:
                    # No sampling for precomputed - use full data or skip
                    metrics['silhouette'] = 0
            
            composite_score = (metrics['silhouette'] + 1) / 2  # Simple normalization
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = (k, labels, metrics)
        except Exception as e:
            st.warning(f"Hierarchical clustering failed for k={k}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / total_iterations)
    
    progress_bar.empty()
    
    if best_model:
        results['Best Model'] = {
            'type': 'Gower + Hierarchical',
            'labels': best_model[1],
            'score': best_score,
            'metrics': best_model[2],
            'params': f"k={best_model[0]}",
            'model': None
        }
    else:
        results['Best Model'] = {'score': -1, 'type': 'Failed'}
    
    return results

# ============================================================
# Anomaly Detection (Optimized)
# ============================================================

def run_anomaly_detection(
    df: pd.DataFrame, 
    features: List[str], 
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Optimized Isolation Forest with better preprocessing"""
    st.write("🔍 Running Isolation Forest anomaly detection...")
    
    df_working = df[features].copy()
    
    # Efficient encoding and handling
    encoders = {}
    for col in df_working.columns:
        if not pd.api.types.is_numeric_dtype(df_working[col]):
            le = LabelEncoder()
            df_working[col] = le.fit_transform(df_working[col].astype(str))
            encoders[col] = le
    
    # Fill missing values with median (more robust than 0)
    df_working = df_working.fillna(df_working.median())
    
    # Fit Isolation Forest
    model = IsolationForest(
        contamination=contamination, 
        random_state=Config.DEFAULT_RANDOM_STATE,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1  # Use all available cores
    )
    
    predictions = model.fit_predict(df_working)
    anomaly_scores = model.score_samples(df_working)
    
    # -1 is anomaly, 1 is normal
    anomalies = df[predictions == -1].copy()
    anomalies['anomaly_score'] = anomaly_scores[predictions == -1]
    
    return anomalies, predictions

# ============================================================
# LLM Integration for Business Insights (MODIFIED FOR HUGGINGFACE)
# ============================================================

def generate_cluster_descriptions(
    df_viz: pd.DataFrame, 
    features: List[str], 
    overall_means: pd.Series,
    api_key: Optional[str] = None,
    threshold: float = 0.15
) -> Tuple[Dict, pd.DataFrame]:
    """
    Generate Cluster Statistics (Table) and AI-powered descriptions (if Key provided).
    Uses HuggingFace Inference API.
    """
    desc_text = {}
    grouped = df_viz[df_viz['Cluster'] != '-1'].groupby('Cluster')
    
    # --- 1. Calculate Cluster Statistics (Table) ---
    stats_data = []
    
    for cluster_id, cluster_data in grouped:
        row = {'Cluster': cluster_id}
        for feat in features:
            if pd.api.types.is_numeric_dtype(df_viz[feat]):
                # Numeric: Mean
                row[feat] = round(cluster_data[feat].mean(), 2)
            else:
                # Categorical: Mode
                modes = cluster_data[feat].mode()
                row[feat] = modes.iloc[0] if not modes.empty else "N/A"
        stats_data.append(row)
    
    cluster_stats_df = pd.DataFrame(stats_data)
    if not cluster_stats_df.empty:
        cluster_stats_df = cluster_stats_df.set_index('Cluster').sort_index()

    # --- 2. Generate AI Descriptions (Only if API Key exists) ---
    if not api_key:
        for cluster_id in grouped.groups.keys():
            desc_text[str(cluster_id)] = {
                "輪廓": "未啟用 AI 分析",
                "特徵": "請提供 HuggingFace API Key 以獲得詳細解讀",
                "策略": "觀察上方視覺化圖表進行人工分析"
            }
        return desc_text, cluster_stats_df
    
    # NEW: HuggingFace API setup
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:        
        for cluster_id, cluster_data in grouped:
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(df_viz)) * 100
            
            # Build feature summary
            feature_summary = []
            for feat in features:
                if pd.api.types.is_numeric_dtype(df_viz[feat]):
                    cluster_val = cluster_data[feat].mean()
                    overall_val = overall_means.get(feat, 0)
                    
                    if overall_val != 0:
                        diff_pct = ((cluster_val - overall_val) / overall_val) * 100
                        if abs(diff_pct) > threshold * 100:
                            direction = "高於" if diff_pct > 0 else "低於"
                            feature_summary.append(
                                f"{feat}: {cluster_val:.2f} ({direction}平均 {abs(diff_pct):.1f}%)"
                            )
                else:
                    # For categorical features, show mode
                    mode_val = cluster_data[feat].mode()
                    if len(mode_val) > 0:
                        feature_summary.append(f"{feat}: 主要為 {mode_val.iloc[0]}")
            
            # Create prompt for Qwen2.5-7B-Instruct
            prompt = f"""
你是一位資深的客戶分群分析專家。請根據以下數據分析這個客戶群組：

群組編號: {cluster_id}
群組大小: {cluster_size} 人 ({cluster_pct:.1f}%)

特徵數據:
{chr(10).join(feature_summary)}

請用繁體中文提供以下三個面向的分析（每個面向 2-3 句話）：

1. 輪廓：這群客戶的核心特徵是什麼？
2. 特徵：這群客戶與其他群組相比有什麼獨特之處？
3. 策略：針對這群客戶應該採取什麼樣的營銷或服務策略？

請以 JSON 格式回應，格式如下：
{{
    "輪廓": "...",
    "特徵": "...",
    "策略": "..."
}}
"""
            # NEW: HuggingFace API Request
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.7,
                    "max_new_tokens": 1000,
                    "return_full_text": False, # Only return the generated text
                    "stop": ["```", "}}"] # Stop sequences to prevent premature termination
                }
            }
            
            response = requests.post(
                Config.HF_API_URL, 
                headers=headers, 
                json=payload
            )
            response.raise_for_status() # Raise exception for bad status codes (e.g., 401, 500)

            # HuggingFace API returns a list of results
            result_list = response.json()
            if not result_list or 'generated_text' not in result_list[0]:
                 raise ValueError("Invalid response format from HuggingFace API.")
            
            response_text = result_list[0]['generated_text'].strip()

            # Parse JSON response
            try:
                # Clean up potential markdown formatting (```json ... ```)
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                
                desc_text[str(cluster_id)] = json.loads(response_text)
            except json.JSONDecodeError:
                desc_text[str(cluster_id)] = {
                    "輪廓": response_text[:200] if len(response_text) > 200 else response_text,
                    "特徵": "AI 分析中...",
                    "策略": "請參考視覺化結果"
                }
                
    except requests.exceptions.RequestException as e:
        # NEW: Handle requests exceptions (API errors, connection issues)
        st.error(f"HuggingFace API 錯誤: {e}")
        for cluster_id in grouped.groups.keys():
            desc_text[str(cluster_id)] = {
                "輪廓": "API 呼叫失敗",
                "特徵": str(e),
                "策略": "請檢查 API Key 或模型是否正確"
            }
    except Exception as e:
        st.error(f"生成描述時發生錯誤: {e}")
        for cluster_id in grouped.groups.keys():
            desc_text[str(cluster_id)] = {
                "輪廓": "分析失敗",
                "特徵": str(e),
                "策略": "請稍後再試"
            }
    
    return desc_text, cluster_stats_df

def generate_anomaly_insights(
    anomalies: pd.DataFrame,
    features: List[str],
    api_key: Optional[str] = None
) -> str:
    """
    Generate AI-powered insights about detected anomalies using HuggingFace API.
    """
    if not api_key or anomalies.empty:
        return "未啟用 AI 分析或無異常資料"
    
    # NEW: HuggingFace API setup
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Sample anomalies for analysis
        sample_size = min(5, len(anomalies))
        anomaly_sample = anomalies[features].head(sample_size)
        
        # Create summary statistics
        anomaly_stats = []
        for feat in features:
            if pd.api.types.is_numeric_dtype(anomalies[feat]):
                anomaly_stats.append(
                    f"{feat}: 範圍 {anomalies[feat].min():.2f} - {anomalies[feat].max():.2f}, "
                    f"平均 {anomalies[feat].mean():.2f}"
                )
        
        prompt = f"""
你是一位資深的數據異常分析專家。請分析以下異常值檢測結果：

偵測到的異常資料數量: {len(anomalies)}
分析特徵: {', '.join(features)}

異常值統計:
{chr(10).join(anomaly_stats)}

請用繁體中文提供：
1. 這些異常值可能代表什麼？（業務角度）
2. 這些異常是否需要關注？為什麼？
3. 建議採取什麼行動？

請以簡潔的段落形式回應（3-5 句話）。
"""
        # NEW: HuggingFace API Request
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 500,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            Config.HF_API_URL, 
            headers=headers, 
            json=payload
        )
        response.raise_for_status()
        
        result_list = response.json()
        if not result_list or 'generated_text' not in result_list[0]:
             return "AI 分析失敗: 無法從 HuggingFace API 獲取有效回應。"

        return result_list[0]['generated_text'].strip()
        
    except requests.exceptions.RequestException as e:
        # NEW: Handle requests exceptions
        return f"AI 分析失敗: HuggingFace API 錯誤: {e}"
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ============================================================
# Visualization (Enhanced)
# ============================================================

def create_cluster_visualization(
    df: pd.DataFrame, 
    features: List[str], 
    labels: np.ndarray,
    title: str = "Clustering Results"
) -> None:
    """Enhanced visualization with better styling"""
    df_viz = df.copy()
    df_viz['Cluster'] = labels.astype(str)
    df_viz_clean = df_viz[df_viz['Cluster'] != '-1']
    
    # Ensure we only use the actual feature columns for plotting
    plot_features = [f for f in features if f in df_viz_clean.columns]
    
    if len(plot_features) < 2:
        st.warning("Need at least 2 valid features for visualization")
        return
    
    if len(plot_features) == 2:
        fig = px.scatter(
            df_viz_clean, 
            x=plot_features[0], 
            y=plot_features[1], 
            color='Cluster',
            title=title,
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
    elif len(plot_features) >= 3:
        # Use only first 3 features for 3D plot
        fig = px.scatter_3d(
            df_viz_clean, 
            x=plot_features[0], 
            y=plot_features[1], 
            z=plot_features[2],  # Use exactly the 3rd feature
            color='Cluster',
            title=title,
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    st.plotly_chart(fig, use_container_width=True)

def display_cluster_statistics(df: pd.DataFrame, labels: np.ndarray) -> None:
    """Display detailed cluster statistics"""
    df_viz = df.copy()
    df_viz['Cluster'] = labels.astype(str)
    
    stats = df_viz['Cluster'].value_counts().reset_index()
    stats.columns = ['Cluster', 'Count']
    stats['Percentage'] = (stats['Count'] / len(df_viz) * 100).round(2)
    stats = stats.sort_values('Cluster')
    
    st.dataframe(stats, use_container_width=True)

# ============================================================
# Main Application
# ============================================================

def main():
    st.title("👥 全方位客戶分群系統 (Advanced Clustering)")
    st.markdown("### 智能客戶分群與異常偵測系統")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📁 1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.divider()
        st.header("⚙️ 2. Clustering Parameters")
        min_c = st.number_input("Minimum clusters", 2, 5, 2)
        max_c = st.number_input("Maximum clusters", 6, 20, 8)
        
        st.divider()
        st.header("🔍 3. Anomaly Detection")
        contamination = st.slider(
            "Contamination ratio", 
            0.01, 0.20, 0.05, 0.01,
            help="Expected proportion of anomalies in dataset"
        )
        
        st.divider()
        st.header("🤖 4. AI Integration (Optional)")
        # MODIFIED: Changed label and variable name for HuggingFace
        hf_api_key = st.text_input("HuggingFace Inference API Key (選填)", type="password")
        if hf_api_key:
             st.caption(f"已輸入 API Key，將使用 {Config.HF_MODEL_NAME} 啟用 AI 洞察功能")
    
    if not uploaded_file:
        st.info("👆 Please upload a CSV file to begin")
        return
    
    # Load data
    try:
        df = load_data(uploaded_file)
    except Exception:
        return
    
    st.subheader("📊 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("View raw data"):
        st.dataframe(df.head(20))
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    # Smart column filtering - more conservative approach
    all_cols = df.columns.tolist()
    
    # Filter out ID columns and columns with too many unique values (likely IDs)
    potential_cols = []
    for c in all_cols:
        # Skip if column name suggests it's an ID
        if 'ID' in c.upper() or c.upper().endswith('_ID'):
            continue
        
        # For categorical columns, skip if too many unique values
        if df[c].dtype == 'object':
            if df[c].nunique() > min(50, len(df) * 0.5):
                continue
        # For numeric columns, skip if unique ratio > 95% (likely continuous ID)
        elif df[c].nunique() == len(df):
            continue
            
        potential_cols.append(c)
    
    # Show column info
    with st.expander("📋 Available columns info"):
        col_info = pd.DataFrame({
            'Column': potential_cols,
            'Type': [str(df[c].dtype) for c in potential_cols],
            'Unique Values': [df[c].nunique() for c in potential_cols],
            'Missing': [df[c].isnull().sum() for c in potential_cols]
        })
        st.dataframe(col_info, use_container_width=True)
    
    selected_features = st.multiselect(
        "Select 2-3 features for clustering (numeric or categorical):",
        potential_cols,
        max_selections=3,
        help="Choose 2-3 features that represent customer behavior or characteristics"
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    # Detect data types
    num_cols, cat_cols = detect_column_types(df, selected_features)
    
    # Determine algorithm
    if len(cat_cols) == 0:
        algo_type = "Numeric (K-Means)"
        algo_color = "🔵"
    elif len(num_cols) == 0:
        algo_type = "Categorical (K-Modes)"
        algo_color = "🟢"
    else:
        algo_type = "Mixed (Gower + Hierarchical)"
        algo_color = "🟣"
    
    st.info(f"{algo_color} **Detected algorithm:** {algo_type}")
    
    # Data preview
    if len(selected_features) in [2, 3]:
        with st.expander("👁️ View feature distribution"):
            df_plot = df[selected_features].dropna()
            
            # Ensure we have data to plot
            if df_plot.empty:
                st.warning("No data available after removing missing values")
            else:
                try:
                    if len(selected_features) == 2:
                        fig = px.scatter(
                            df_plot, x=selected_features[0], y=selected_features[1],
                            title="2D Distribution Preview",
                            template='plotly_white'
                        )
                    else:  # len == 3
                        fig = px.scatter_3d(
                            df_plot, 
                            x=selected_features[0], 
                            y=selected_features[1], 
                            z=selected_features[2],
                            title="3D Distribution Preview",
                            template='plotly_white'
                        )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating preview: {e}")
                    st.info("This might be due to incompatible data types. Try selecting different features.")
    
    # Action buttons
    col_cluster, col_anomaly = st.columns(2)
    
    with col_cluster:
        run_clustering = st.button(
            "🚀 Start Clustering", 
            type="primary",
            use_container_width=True
        )
    
    with col_anomaly:
        run_anomaly = st.button(
            "🔍 Detect Anomalies",
            use_container_width=True
        )
    
    # Execute clustering
    if run_clustering:
        with st.spinner("🔄 Processing data and running clustering algorithms..."):
            try:
                df_used = df[selected_features].dropna().copy()
                
                if df_used.empty:
                    st.error("No valid data after removing missing values")
                    return
                
                # Store in session state
                st.session_state.features = selected_features
                st.session_state.df_used = df_used
                
                # Run appropriate algorithm
                if algo_type == "Numeric (K-Means)":
                    X_scaled, transform_info, _ = smart_preprocessing_numeric(
                        df, selected_features
                    )
                    results = run_kmeans(X_scaled, (min_c, max_c))
                    
                elif algo_type == "Categorical (K-Modes)":
                    results = run_kmodes(df_used, (min_c, max_c))
                    
                elif algo_type == "Mixed (Gower + Hierarchical)":
                    results = run_gower_hierarchical(df_used, (min_c, max_c))
                
                st.session_state.results = results
                st.session_state.ran_clustering = True
                
            except Exception as e:
                st.error(f"Error during clustering: {e}")
                return
    
    # Display clustering results
    if st.session_state.get('ran_clustering'):
        st.divider()
        st.header("🎯 Clustering Results")
        
        results = st.session_state.results
        best_res = results.get('Best Model')
        
        if best_res and best_res['score'] > 0:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = best_res.get('metrics', {})
            
            col1.metric("Model", best_res['type'])
            col2.metric("Composite Score", f"{best_res['score']:.4f}")
            col3.metric("Silhouette", f"{metrics.get('silhouette', -1):.4f}")
            col4.metric("Clusters", metrics.get('n_clusters', 0))
            
            # Visualization
            st.subheader("📈 Cluster Visualization")
            create_cluster_visualization(
                st.session_state.df_used,
                selected_features,
                best_res['labels'],
                f"{best_res['type']} - {best_res['params']}"
            )
            
            # Statistics
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Cluster Statistics**")
                display_cluster_statistics(
                    st.session_state.df_used,
                    best_res['labels']
                )
            
            with col2:
                st.write("**Quality Metrics**")
                if metrics:
                    metrics_df = pd.DataFrame([{
                        'Metric': k.replace('_', ' ').title(),
                        'Value': f"{v:.4f}" if isinstance(v, float) else v
                    } for k, v in metrics.items()])
                    st.dataframe(metrics_df, use_container_width=True)
            
            # -----------------------------------------------------------------
            # Cluster Statistics Table (Mean/Mode) - Modified Section
            # -----------------------------------------------------------------
            st.divider()
            st.subheader("📋 各群組特徵統計 (Cluster Feature Statistics)")
            st.caption("數值特徵顯示**平均值 (Mean)**，類別特徵顯示**眾數 (Mode)**")

            df_viz = st.session_state.df_used.copy()
            df_viz['Cluster'] = best_res['labels'].astype(str)
            
            # Calculate overall means (only used for LLM prompt if needed)
            numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])]
            overall_means = df[numeric_features].mean() if numeric_features else pd.Series()
            
            # Generate descriptions AND the table (Passing hf_api_key)
            descriptions, cluster_stats_df = generate_cluster_descriptions(
                df_viz, selected_features, overall_means, hf_api_key
            )

            # Display the statistics table
            if not cluster_stats_df.empty:
                st.dataframe(cluster_stats_df.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)

            # -----------------------------------------------------------------
            # AI-Powered Business Insights (Conditional)
            # -----------------------------------------------------------------
            st.subheader("💡 AI 業務洞察分析")
            
            if hf_api_key:
                # Display descriptions if API key is provided
                for cluster_id, desc in descriptions.items():
                    if cluster_id != '-1':
                        with st.expander(f"📊 群組 {cluster_id} 分析", expanded=True):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.markdown("**👤 客戶輪廓**")
                                st.write(desc.get('輪廓', 'N/A'))
                            
                            with col_b:
                                st.markdown("**🔍 獨特特徵**")
                                st.write(desc.get('特徵', 'N/A'))
                            
                            with col_c:
                                st.markdown("**💼 營銷策略**")
                                st.write(desc.get('策略', 'N/A'))
            else:
                st.info("💡 提供 HuggingFace Inference API Key 以啟用 AI 智能分析功能")
                st.markdown(f"""
                **AI 分析功能包括：** (使用模型: **{Config.HF_MODEL_NAME}**)
                - 🎯 自動識別各群組的核心特徵
                - 📊 比較不同群組的差異
                - 💼 提供針對性的營銷策略建議
                
                [了解 HuggingFace Inference API](https://huggingface.co/docs/api-inference/index) →
                """)
        else:
            st.error("❌ Clustering failed. Please try different parameters or features.")
    
    # Execute anomaly detection
    if run_anomaly:
        with st.spinner("🔍 Detecting anomalies..."):
            try:
                anomalies, predictions = run_anomaly_detection(
                    df, selected_features, contamination
                )
                st.session_state.anomalies = anomalies
                st.session_state.anomaly_predictions = predictions
                st.session_state.ran_anomaly = True
            except Exception as e:
                st.error(f"Error during anomaly detection: {e}")
    
    # Display anomaly results
    if st.session_state.get('ran_anomaly'):
        st.divider()
        st.header("⚠️ Anomaly Detection Results")
        
        anomalies = st.session_state.anomalies
        predictions = st.session_state.anomaly_predictions
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomalies", len(anomalies))
        col2.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")
        col3.metric("Normal Samples", len(df) - len(anomalies))
        
        if not anomalies.empty:
            st.subheader("🔴 Detected Anomalies")
            st.dataframe(anomalies.head(20))
            
            # AI-Powered Anomaly Insights (Passing hf_api_key)
            if hf_api_key:
                st.markdown("### 🤖 AI 異常分析")
                with st.spinner("正在分析異常資料..."):
                    insights = generate_anomaly_insights(
                        anomalies, selected_features, hf_api_key
                    )
                    st.info(insights)
            
            # Visualization
            df_plot = df[selected_features].copy()
            df_plot['Type'] = np.where(predictions == -1, 'Anomaly', 'Normal')
            
            # Only plot if we have valid features
            plot_features = [f for f in selected_features if f in df_plot.columns]
            
            try:
                if len(plot_features) == 2:
                    fig = px.scatter(
                        df_plot, 
                        x=plot_features[0], 
                        y=plot_features[1],
                        color='Type',
                        color_discrete_map={'Normal': '#E8E8E8', 'Anomaly': '#FF4444'},
                        title="Anomaly Distribution",
                        template='plotly_white'
                    )
                elif len(plot_features) >= 3:
                    fig = px.scatter_3d(
                        df_plot,
                        x=plot_features[0],
                        y=plot_features[1],
                        z=plot_features[2],
                        color='Type',
                        color_discrete_map={'Normal': '#E8E8E8', 'Anomaly': '#FF4444'},
                        title="Anomaly Distribution (3D)",
                        template='plotly_white'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating anomaly visualization: {e}")

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # Initialize session state
    if 'ran_clustering' not in st.session_state:
        st.session_state.ran_clustering = False
    if 'ran_anomaly' not in st.session_state:
        st.session_state.ran_anomaly = False
    
    # Check dependencies
    if not ADVANCED_AVAILABLE:
        st.warning(
            "⚠️ Advanced clustering features (K-Modes, Gower) unavailable. "
            "Install with: `pip install kmodes gower`"
        )
    
    main()
