import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

# 字体设置函数
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试加载项目中的字体文件
        font_path = 'simhei.ttf'  # 字体文件应该在项目根目录
        
        if os.path.exists(font_path):
            # 使用自定义字体
            chinese_font = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = chinese_font.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return chinese_font
        else:
            # 如果字体文件不存在，尝试系统字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
            plt.rcParams['axes.unicode_minus'] = False
            return None
    except Exception as e:
        st.warning(f"字体加载失败，使用默认字体: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return None

# 获取中文字体属性
def get_chinese_font_prop():
    """获取中文字体属性用于matplotlib"""
    font_path = 'simhei.ttf'
    if os.path.exists(font_path):
        return FontProperties(fname=font_path)
    else:
        return FontProperties()

class CarbonEmissionAssessment:
    def __init__(self):
        self.emission_factors = {
            'electricity': 0.5810,
            'natural_gas': 2.162,
            'coal': 2.493,
            'diesel': 2.674,
            'gasoline': 2.296,
            'steam': 0.126,
            'water': 0.91,
        }
        self.time_factors = {
            'equipment_operation': 5.2,
            'transportation': 12.5,
            'labor_indirect': 2.1,
        }
        self.custom_factors = {}
        self.factor_correlations = {}

    def set_custom_factors(self, factors_config):
        self.custom_factors = factors_config
        for factor_name, config in factors_config.items():
            self.factor_correlations[factor_name] = config.get('correlation', 'positive')

    def calculate_entropy_weights(self, data, correlation_settings=None):
        data_processed = data.copy()
        if correlation_settings:
            for col, correlation in correlation_settings.items():
                if col in data_processed.columns and correlation == 'negative':
                    max_val = data_processed[col].max()
                    min_val = data_processed[col].min()
                    if max_val != min_val:
                        data_processed[col] = max_val + min_val - data_processed[col]
        data_normalized = (data_processed - data_processed.min()) / (data_processed.max() - data_processed.min())
        data_normalized = data_normalized.fillna(0).replace(0, 1e-10)
        m, n = data_normalized.shape
        entropy = np.zeros(n)
        for j in range(n):
            p = data_normalized.iloc[:, j] / data_normalized.iloc[:, j].sum()
            p = p.replace(0, 1e-10)
            entropy[j] = -1 / np.log(m) * np.sum(p * np.log(p))
        weights = (1 - entropy) / np.sum(1 - entropy)
        return weights, correlation_settings

    def calculate_carbon_emission_basic(self, data_row):
        total_emission = 0
        emission_details = {}
        if 'electricity_kwh' in data_row and pd.notna(data_row['electricity_kwh']):
            electricity_emission = data_row['electricity_kwh'] * self.emission_factors['electricity']
            total_emission += electricity_emission
            emission_details['电力'] = electricity_emission
        if 'natural_gas_m3' in data_row and pd.notna(data_row['natural_gas_m3']):
            gas_emission = data_row['natural_gas_m3'] * self.emission_factors['natural_gas']
            total_emission += gas_emission
            emission_details['天然气'] = gas_emission
        if 'diesel_l' in data_row and pd.notna(data_row['diesel_l']):
            diesel_emission = data_row['diesel_l'] * self.emission_factors['diesel']
            total_emission += diesel_emission
            emission_details['柴油'] = diesel_emission
        if 'operation_hours' in data_row and pd.notna(data_row['operation_hours']):
            time_emission = data_row['operation_hours'] * self.time_factors['equipment_operation']
            total_emission += time_emission
            emission_details['设备运行时间'] = time_emission
        if 'transport_hours' in data_row and pd.notna(data_row['transport_hours']):
            transport_emission = data_row['transport_hours'] * self.time_factors['transportation']
            total_emission += transport_emission
            emission_details['运输时间'] = transport_emission
        return total_emission, emission_details

    def calculate_carbon_emission_custom(self, data_row, custom_columns, weights):
        total_emission, emission_details = self.calculate_carbon_emission_basic(data_row)
        if len(custom_columns) > 0 and weights is not None:
            custom_emission = 0
            for i, col in enumerate(custom_columns):
                if col in data_row and pd.notna(data_row[col]):
                    factor_config = self.custom_factors.get(col, {})
                    emission_factor = factor_config.get('emission_factor', 1.0)
                    factor_emission = data_row[col] * emission_factor * weights[i]
                    custom_emission += factor_emission
                    emission_details[f'自定义-{col}'] = factor_emission
            total_emission += custom_emission
        return total_emission, emission_details

    def assess_multiple_processes(self, df, custom_columns=None, correlation_settings=None):
        if custom_columns is None:
            custom_columns = []
        if len(custom_columns) == 0:
            indicator_columns = ['electricity_cost', 'fuel_cost', 'operation_hours', 'production_volume']
            available_columns = [col for col in indicator_columns if col in df.columns]
        else:
            available_columns = [col for col in custom_columns if col in df.columns]
        if len(available_columns) < 2:
            st.error("至少需要2个评估指标才能进行熵权法分析")
            return None, None, None, None
        weights, processed_correlations = self.calculate_entropy_weights(df[available_columns], correlation_settings)
        emissions = []
        emission_details_list = []
        for idx, row in df.iterrows():
            if len(custom_columns) > 0:
                emission, details = self.calculate_carbon_emission_custom(row, available_columns, weights)
            else:
                emission, details = self.calculate_carbon_emission_basic(row)
            emissions.append(emission)
            emission_details_list.append(details)
        df['carbon_emission'] = emissions
        weighted_scores = np.zeros(len(df))
        for i, col in enumerate(available_columns):
            weight_sign = 1
            if correlation_settings and col in correlation_settings:
                if correlation_settings[col] == 'negative':
                    weight_sign = -1
            weighted_scores += df[col] * weights[i] * weight_sign
        df['weighted_score'] = weighted_scores
        return df, weights, emission_details_list, available_columns

def create_custom_factors_interface():
    st.sidebar.header("🎛️ 自定义评估要素")
    num_factors = st.sidebar.number_input("评估要素数量", min_value=2, max_value=10, value=4)
    custom_factors = {}
    correlation_settings = {}
    custom_columns = []
    for i in range(num_factors):
        st.sidebar.subheader(f"要素 {i+1}")
        factor_name = st.sidebar.text_input(f"要素名称 {i+1}", value=f"factor_{i+1}", key=f"factor_name_{i}")
        emission_factor = st.sidebar.number_input(f"排放因子 {i+1} (kg CO2/单位)", value=1.0, step=0.1, key=f"emission_factor_{i}")
        correlation = st.sidebar.selectbox(f"与碳排放相关性 {i+1}", ["positive", "negative"], format_func=lambda x: "正相关 (数值越大，排放越大)" if x == "positive" else "负相关 (数值越大，排放越小)", key=f"correlation_{i}")
        custom_factors[factor_name] = {'emission_factor': emission_factor, 'correlation': correlation}
        correlation_settings[factor_name] = correlation
        custom_columns.append(factor_name)
    return custom_factors, correlation_settings, custom_columns

def manual_input_table_interface(assessment, assessment_mode, custom_columns=None, correlation_settings=None):
    st.subheader("📝 手动输入多个流程数据（可直接编辑）")
    if assessment_mode == "预设模式 (电费、燃料费等)":
        default_data = [
            {"流程名称": "流程A", "电力消耗 (kWh)": 120, "电费 (元)": 78, "天然气消耗 (m³)": 25,
             "燃料费 (元)": 75, "运行时间 (小时)": 10, "运输时间 (小时)": 3, "产量 (单位)": 1200, "柴油消耗 (L)": 15},
            {"流程名称": "流程B", "电力消耗 (kWh)": 85, "电费 (元)": 55, "天然气消耗 (m³)": 15,
             "燃料费 (元)": 45, "运行时间 (小时)": 6, "运输时间 (小时)": 2, "产量 (单位)": 800, "柴油消耗 (L)": 8},
            {"流程名称": "流程C", "电力消耗 (kWh)": 200, "电费 (元)": 130, "天然气消耗 (m³)": 40,
             "燃料费 (元)": 120, "运行时间 (小时)": 16, "运输时间 (小时)": 5, "产量 (单位)": 2000, "柴油消耗 (L)": 25},
        ]
        df_manual = pd.DataFrame(default_data)
    else:
        if not custom_columns:
            st.error("请先配置自定义要素")
            return None
        default_data = []
        for i in range(3):
            row = {"流程名称": f"流程{chr(65+i)}"}
            for col in custom_columns:
                corr_text = " (正相关)" if correlation_settings[col] == 'positive' else " (负相关)"
                row[col + corr_text] = 100.0
            default_data.append(row)
        df_manual = pd.DataFrame(default_data)
    edited_df = st.data_editor(df_manual, num_rows="dynamic", use_container_width=True)
    if assessment_mode == "预设模式 (电费、燃料费等)":
        edited_df.columns = [
            'process_name', 'electricity_kwh', 'electricity_cost', 'natural_gas_m3',
            'fuel_cost', 'operation_hours', 'transport_hours', 'production_volume', 'diesel_l'
        ]
    else:
        edited_df.columns = ['process_name'] + custom_columns
    return edited_df

def display_results(df, weights, details_list, used_columns, assessment, custom_factors=None, correlation_settings=None):
    st.header("📊 分析结果")
    
    # 获取中文字体属性
    font_prop = get_chinese_font_prop()
    
    if 'production_volume' in df.columns:
        df['emission_intensity'] = df['carbon_emission'] / df['production_volume']
        result_columns = ['process_name', 'carbon_emission', 'emission_intensity', 'weighted_score']
        column_names = ['流程名称', '总碳排放(kg CO2)', '排放强度(kg CO2/单位)', '加权评分']
    else:
        result_columns = ['process_name', 'carbon_emission', 'weighted_score']
        column_names = ['流程名称', '总碳排放(kg CO2)', '加权评分']
    result_df = df[result_columns].copy()
    result_df.columns = column_names
    result_df = result_df.sort_values('总碳排放(kg CO2)')
    st.dataframe(result_df.round(3))
    
    st.subheader("⚖️ 熵权法计算权重")
    weights_data = []
    for i, col in enumerate(used_columns):
        correlation_type = "正相关"
        if correlation_settings and col in correlation_settings:
            correlation_type = "正相关" if correlation_settings[col] == "positive" else "负相关"
        emission_factor = "系统默认"
        if custom_factors and col in custom_factors:
            emission_factor = f"{custom_factors[col]['emission_factor']:.3f}"
        weights_data.append({
            '评估指标': col,
            '权重': weights[i],
            '权重百分比': f"{weights[i] * 100:.2f}%",
            '相关性': correlation_type,
            '排放因子': emission_factor
        })
    weights_df = pd.DataFrame(weights_data)
    st.dataframe(weights_df)
    
    st.subheader("📈 可视化分析")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['process_name'], df['carbon_emission'], color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df))))
        ax.set_title('各流程碳排放对比', fontproperties=font_prop)
        ax.set_ylabel('碳排放量 (kg CO2)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        # 设置x轴标签字体
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}', ha='center', va='bottom', fontproperties=font_prop)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        labels_with_correlation = []
        for col in used_columns:
            correlation_type = "+"
            if correlation_settings and col in correlation_settings:
                correlation_type = "+" if correlation_settings[col] == "positive" else "-"
            labels_with_correlation.append(f"{col}({correlation_type})")
        wedges, texts, autotexts = ax.pie(weights, labels=labels_with_correlation, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('熵权法 - 各指标权重分布\n(+正相关, -负相关)', fontproperties=font_prop)
        # 设置饼图标签字体
        for text in texts:
            text.set_fontproperties(font_prop)
        for autotext in autotexts:
            autotext.set_fontproperties(font_prop)
        st.pyplot(fig)
    
    if 'emission_intensity' in df.columns:
        st.subheader("📊 排放强度分析")
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(df['process_name'], df['emission_intensity'], color=plt.cm.Blues(np.linspace(0.3, 0.8, len(df))))
        ax.set_title('各流程碳排放强度对比', fontproperties=font_prop)
        ax.set_ylabel('排放强度 (kg CO2/单位产品)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        # 设置坐标轴标签字体
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontproperties=font_prop)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("🎯 综合评分分析")
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = df.sort_values('weighted_score', ascending=True)
    bars = ax.barh(sorted_df['process_name'], sorted_df['weighted_score'], color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df))))
    ax.set_title('各流程综合评分对比 (考虑权重和相关性)', fontproperties=font_prop)
    ax.set_xlabel('加权综合评分', fontproperties=font_prop)
    # 设置坐标轴标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_prop)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_prop)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontproperties=font_prop)
    plt.tight_layout()
    st.pyplot(fig)
    
    if custom_factors and correlation_settings:
        st.subheader("🔍 要素相关性分析")
        factor_columns = [col for col in used_columns if col in df.columns]
        if len(factor_columns) > 1:
            corr_matrix = df[factor_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f', ax=ax)
            ax.set_title('评估要素相关性矩阵', fontproperties=font_prop)
            # 设置坐标轴标签字体
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(font_prop)
            for tick in ax.get_yticklabels():
                tick.set_fontproperties(font_prop)
            st.pyplot(fig)
        
        st.subheader("💪 要素影响力分析")
        influence_data = []
        for i, col in enumerate(used_columns):
            if col in df.columns:
                correlation_type = correlation_settings.get(col, 'positive')
                weight = weights[i]
                std_dev = df[col].std()
                influence_score = weight * std_dev
                influence_data.append({
                    '要素名称': col,
                    '权重': weight,
                    '标准差': std_dev,
                    '影响力得分': influence_score,
                    '相关性': '正相关' if correlation_type == 'positive' else '负相关'
                })
        influence_df = pd.DataFrame(influence_data)
        influence_df = influence_df.sort_values('影响力得分', ascending=False)
        st.dataframe(influence_df.round(4))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if corr == '正相关' else 'red' for corr in influence_df['相关性']]
        bars = ax.bar(influence_df['要素名称'], influence_df['影响力得分'], color=colors, alpha=0.7)
        ax.set_title('各要素影响力得分 (绿色:正相关, 红色:负相关)', fontproperties=font_prop)
        ax.set_ylabel('影响力得分 (权重 × 标准差)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        # 设置坐标轴标签字体
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontproperties=font_prop)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("💡 改进建议")
    max_emission_idx = df['carbon_emission'].idxmax()
    min_emission_idx = df['carbon_emission'].idxmin()
    max_process = df.loc[max_emission_idx, 'process_name']
    min_process = df.loc[min_emission_idx, 'process_name']
    max_emission = df.loc[max_emission_idx, 'carbon_emission']
    min_emission = df.loc[min_emission_idx, 'carbon_emission']
    improvement_potential = max_emission - min_emission
    st.info(f"""
    **🌟 最优流程**: {min_process} (碳排放: {min_emission:.1f} kg CO2)
    
    **⚠️ 需改进流程**: {max_process} (碳排放: {max_emission:.1f} kg CO2)
    
    **🎯 改进潜力**: 如果将最高排放流程优化到最佳水平，可减少 {improvement_potential:.1f} kg CO2 排放
    """)
    
    if len(used_columns) > 0:
        st.subheader("🔧 具体改进建议")
        weight_importance = list(zip(used_columns, weights))
        weight_importance.sort(key=lambda x: x[1], reverse=True)
        top_factors = weight_importance[:3]
        suggestions = []
        for factor, weight in top_factors:
            correlation_type = "正相关"
            if correlation_settings and factor in correlation_settings:
                correlation_type = "正相关" if correlation_settings[factor] == "positive" else "负相关"
            if correlation_type == "正相关":
                suggestion = f"降低 **{factor}** 的数值"
            else:
                suggestion = f"提高 **{factor}** 的数值"
            suggestions.append(f"- {suggestion} (权重: {weight:.3f})")
        st.markdown("**优先改进建议** (按权重排序):")
        for suggestion in suggestions:
            st.markdown(suggestion)
    
    if len(df) > 1 and len(used_columns) > 1:
        st.subheader("📈 敏感性分析")
        sensitivity_data = []
        for col in used_columns:
            if col in df.columns:
                col_range = df[col].max() - df[col].min()
                emission_range = df['carbon_emission'].max() - df['carbon_emission'].min()
                if col_range > 0:
                    sensitivity = emission_range / col_range
                    sensitivity_data.append({
                        '要素': col,
                        '敏感性': sensitivity,
                        '变化范围': col_range
                    })
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df = sensitivity_df.sort_values('敏感性', ascending=False)
            st.write("**敏感性排序** (数值越大表示该要素对碳排放影响越大):")
            st.dataframe(sensitivity_df.round(4))
    
    st.subheader("📥 结果导出")
    col1, col2 = st.columns(2)
    
    # 准备详细分析结果数据
    export_df = df.copy()
    export_df['评估模式'] = "自定义模式" if custom_factors else "预设模式"
    for i, col in enumerate(used_columns):
        if col in export_df.columns:
            export_df[f'{col}_权重'] = weights[i]
    result_csv = export_df.to_csv(index=False).encode('utf-8-sig')
    
    # 准备权重配置数据
    config_data = []
    for i, col in enumerate(used_columns):
        config_data.append({
            '要素名称': col,
            '权重': weights[i],
            '权重百分比': f"{weights[i] * 100:.2f}%",
            '相关性': correlation_settings.get(col, '正相关') if correlation_settings else '正相关',
            '排放因子': custom_factors.get(col, {}).get('emission_factor', '系统默认') if custom_factors else '系统默认'
        })
    config_df = pd.DataFrame(config_data)
    config_csv = config_df.to_csv(index=False).encode('utf-8-sig')
    
    with col1:
        st.download_button(
            label="📊 下载详细分析结果",
            data=result_csv,
            file_name=f"碳排放评估详细结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="下载包含碳排放计算、权重信息的完整数据",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📋 下载权重配置",
            data=config_csv,
            file_name=f"评估权重配置_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="下载评估要素的权重配置信息",
            use_container_width=True
        )
    
    # 添加数据预览
    with st.expander("🔍 导出数据预览"):
        tab1, tab2 = st.tabs(["详细分析结果", "权重配置"])
        
        with tab1:
            st.write("**详细分析结果数据预览:**")
            preview_df = export_df.copy()
            # 只显示前几列和重要的列
            important_cols = ['process_name', 'carbon_emission', 'weighted_score', '评估模式']
            if 'emission_intensity' in preview_df.columns:
                important_cols.insert(2, 'emission_intensity')
            
            display_cols = important_cols + [col for col in preview_df.columns if col.endswith('_权重')]
            st.dataframe(preview_df[display_cols].round(4), use_container_width=True)
        
        with tab2:
            st.write("**权重配置数据预览:**")
            st.dataframe(config_df, use_container_width=True)
    
    # 添加导出说明
    st.info("""
    **📋 导出文件说明:**
    
    **详细分析结果** 包含:
    - 各流程的碳排放计算结果
    - 排放强度数据 (如适用)
    - 综合评分和权重信息
    - 原始输入数据
    
    **权重配置** 包含:
    - 各评估要素的权重值
    - 权重百分比分布
    - 要素相关性设置
    - 排放因子配置
    """)

def main():
    st.set_page_config(page_title="企业碳排放评估系统", layout="wide")
    
    # 设置中文字体
    setup_chinese_font()
    
    st.title("🌱 基于熵权法的企业碳排放评估系统")
    st.markdown("通过自定义评估要素，快速评估生产流程碳排放")
    
    # 字体状态显示
    font_path = 'simhei.ttf'
    if os.path.exists(font_path):
        st.success("✅ 中文字体已加载，图表将正确显示中文")
    else:
        st.warning("⚠️ 未找到simhei.ttf字体文件，请确保字体文件在项目根目录")
    
    assessment = CarbonEmissionAssessment()
    st.header("🎯 评估模式选择")
    assessment_mode = st.radio(
        "选择评估模式:",
        ["预设模式 (电费、燃料费等)", "自定义模式 (用户定义要素)"],
        help="预设模式使用系统内置的评估要素；自定义模式允许您完全自定义评估要素和相关性"
    )
    custom_factors = {}
    correlation_settings = {}
    custom_columns = []
    if assessment_mode == "自定义模式 (用户定义要素)":
        custom_factors, correlation_settings, custom_columns = create_custom_factors_interface()
        assessment.set_custom_factors(custom_factors)
        with st.expander("📋 当前要素配置"):
            config_df = pd.DataFrame([
                {
                    '要素名称': name,
                    '排放因子': config['emission_factor'],
                    '相关性': '正相关' if config['correlation'] == 'positive' else '负相关'
                }
                for name, config in custom_factors.items()
            ])
            st.dataframe(config_df)
    if assessment_mode == "预设模式 (电费、燃料费等)":
        st.sidebar.header("⚙️ 系统参数设置")
        st.sidebar.subheader("排放因子调整")
        electricity_factor = st.sidebar.number_input("电力排放因子 (kg CO2/kWh)", value=0.5810, step=0.01)
        gas_factor = st.sidebar.number_input("天然气排放因子 (kg CO2/m³)", value=2.162, step=0.01)
        assessment.emission_factors['electricity'] = electricity_factor
        assessment.emission_factors['natural_gas'] = gas_factor
    
    st.header("📊 数据输入")
    input_method = st.radio("选择数据输入方式:", ["使用示例数据", "批量上传CSV文件", "手动输入多个流程"])
    df = None
    
    if input_method == "使用示例数据":
        if assessment_mode == "自定义模式 (用户定义要素)":
            def create_sample_data_with_custom_factors(custom_columns, correlation_settings):
                np.random.seed(42)
                sample_data = {'process_name': [f'流程{chr(65+i)}' for i in range(5)]}
                for i, col in enumerate(custom_columns):
                    if correlation_settings[col] == 'positive':
                        base_values = [50 + i*20 + np.random.normal(0, 10) for i in range(5)]
                    else:
                        base_values = [150 - i*20 + np.random.normal(0, 10) for i in range(5)]
                    sample_data[col] = [max(10, val) for val in base_values]
                return pd.DataFrame(sample_data)
            
            df = create_sample_data_with_custom_factors(custom_columns, correlation_settings)
            st.subheader("📋 自定义示例数据")
            st.dataframe(df.round(2))
            st.info("""
            **示例数据说明**：
            - 正相关要素：数值越大，预期碳排放越大
            - 负相关要素：数值越大，预期碳排放越小
            - 数据已根据相关性设置生成相应趋势
            """)
        else:
            sample_data = {
                'process_name': ['流程A', '流程B', '流程C', '流程D', '流程E'],
                'electricity_kwh': [120, 85, 200, 95, 150],
                'electricity_cost': [78, 55, 130, 62, 98],
                'natural_gas_m3': [25, 15, 40, 18, 30],
                'fuel_cost': [75, 45, 120, 54, 90],
                'operation_hours': [10, 6, 16, 8, 12],
                'transport_hours': [3, 2, 5, 2.5, 4],
                'production_volume': [1200, 800, 2000, 900, 1500],
                'diesel_l': [15, 8, 25, 10, 18]
            }
            df = pd.DataFrame(sample_data)
            st.subheader("📋 预设示例数据")
            st.dataframe(df)
    
    elif input_method == "批量上传CSV文件":
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("📋 上传的数据")
                st.dataframe(df)
            except Exception as e:
                st.error(f"文件读取错误: {e}")
    
    else:
        df = manual_input_table_interface(assessment, assessment_mode, custom_columns, correlation_settings)
    
    if df is not None and not df.empty:
        if st.button("🚀 开始分析"):
            if assessment_mode == "自定义模式 (用户定义要素)":
                analyzed_df, weights, details_list, used_columns = assessment.assess_multiple_processes(
                    df, custom_columns, correlation_settings)
            else:
                analyzed_df, weights, details_list, used_columns = assessment.assess_multiple_processes(df)
            
            if analyzed_df is not None:
                display_results(analyzed_df, weights, details_list, used_columns,
                              assessment,
                              custom_factors if assessment_mode == "自定义模式 (用户定义要素)" else None,
                              correlation_settings if assessment_mode == "自定义模式 (用户定义要素)" else None)
    
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 系统功能
        - **预设模式**: 使用电费、燃料费、运行时间等常见指标
        - **自定义模式**: 完全自定义评估要素和相关性设置
        - **熵权法权重计算**: 客观确定各评估因子的权重
        - **正负相关性处理**: 支持正相关和负相关要素的权重计算
        
        ### 相关性说明
        - **正相关**: 指标数值越大，预期碳排放越大（如：用电量、燃料费）
        - **负相关**: 指标数值越大，预期碳排放越小（如：设备效率、产品质量）
        
        ### 自定义模式CSV格式示例
        ```
        process_name,factor_1,factor_2,factor_3,factor_4
        流程A,120.5,78.2,25.1,95.3
        流程B,85.7,55.9,15.2,88.7
        ```
        
        ### 预设模式CSV格式示例  
        ```
        process_name,electricity_kwh,electricity_cost,natural_gas_m3,fuel_cost,operation_hours
        流程A,120,78,25,75,10
        流程B,85,55,15,45,6
        ```
        
        ### 字体文件配置说明
        为了在Streamlit Cloud上正确显示中文：
        1. 将 `simhei.ttf` 字体文件放在项目根目录
        2. 在 `requirements.txt` 中添加必要的依赖
        3. 系统会自动检测并加载字体文件
        
        ### requirements.txt 示例
        ```
        streamlit
        pandas
        numpy
        matplotlib
        seaborn
        ```
        """)

if __name__ == "__main__":
    main()




