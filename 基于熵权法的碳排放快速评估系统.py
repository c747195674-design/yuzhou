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
            'electricity': 0.6205,
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

    def calculate_coefficient_of_variation_weights(self, data, correlation_settings=None):
        """
        使用变异系数法计算权重，适用于小样本情况
        """
        data_processed = data.copy()
        
        # 处理相关性设置
        if correlation_settings:
            for col, correlation in correlation_settings.items():
                if col in data_processed.columns and correlation == 'negative':
                    max_val = data_processed[col].max()
                    min_val = data_processed[col].min()
                    if max_val != min_val:
                        data_processed[col] = max_val + min_val - data_processed[col]
        
        # 计算变异系数
        cv_values = []
        for col in data_processed.columns:
            mean_val = data_processed[col].mean()
            std_val = data_processed[col].std()
            if mean_val != 0 and std_val != 0:
                cv = std_val / mean_val
            else:
                cv = 0.01  # 避免除零错误
            cv_values.append(abs(cv))
        
        # 标准化变异系数得到权重
        cv_sum = sum(cv_values)
        if cv_sum == 0:
            weights = np.ones(len(cv_values)) / len(cv_values)
        else:
            weights = np.array(cv_values) / cv_sum
        
        return weights, correlation_settings

    def calculate_enhanced_entropy_weights(self, data, correlation_settings=None):
        """
        增强版熵权法，对小样本进行特别处理
        """
        data_processed = data.copy()
        m, n = data_processed.shape
        
        # 处理相关性设置
        if correlation_settings:
            for col, correlation in correlation_settings.items():
                if col in data_processed.columns and correlation == 'negative':
                    max_val = data_processed[col].max()
                    min_val = data_processed[col].min()
                    if max_val != min_val:
                        data_processed[col] = max_val + min_val - data_processed[col]
        
        # 小样本特别处理
        if m <= 3:
            st.info(f"检测到小样本数据({m}个样本)，自动切换到变异系数法计算权重")
            return self.calculate_coefficient_of_variation_weights(data, correlation_settings)
        
        # 标准化处理
        data_normalized = (data_processed - data_processed.min()) / (data_processed.max() - data_processed.min())
        data_normalized = data_normalized.fillna(0)
        
        # 避免0值，使用更小的替代值
        epsilon = 1e-12
        data_normalized = data_normalized.replace(0, epsilon)
        
        # 计算熵值
        entropy = np.zeros(n)
        for j in range(n):
            col_sum = data_normalized.iloc[:, j].sum()
            if col_sum > 0:
                p = data_normalized.iloc[:, j] / col_sum
                p = p.replace(0, epsilon)
                # 增强熵值计算的稳定性
                log_p = np.log(p)
                entropy[j] = -1 / np.log(m) * np.sum(p * log_p)
            else:
                entropy[j] = 1.0
        
        # 计算权重
        entropy_diff = 1 - entropy
        weights_sum = np.sum(entropy_diff)
        
        if weights_sum == 0:
            # 如果所有权重都相同，使用变异系数法作为备选
            st.warning("熵权法计算出现异常，自动切换到变异系数法")
            return self.calculate_coefficient_of_variation_weights(data, correlation_settings)
        
        weights = entropy_diff / weights_sum
        
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
            
        # 选择用于权重计算的列
        if len(custom_columns) == 0:
            indicator_columns = ['electricity_cost', 'fuel_cost', 'operation_hours', 'production_volume']
            available_columns = [col for col in indicator_columns if col in df.columns]
        else:
            available_columns = [col for col in custom_columns if col in df.columns]
        
        if len(available_columns) < 2:
            st.error("至少需要2个评估指标才能进行熵权法分析")
            return None, None, None, None
        
        # 使用增强版熵权法计算权重
        weights, processed_correlations = self.calculate_enhanced_entropy_weights(
            df[available_columns], correlation_settings)
        
        # 显示权重计算方法
        sample_count = len(df)
        if sample_count <= 3:
            st.info(f"样本数量: {sample_count}，已使用变异系数法确保权重差异化")
        else:
            st.info(f"样本数量: {sample_count}，使用熵权法计算权重")
        
        # 计算碳排放
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
        
        # 计算加权得分
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
        factor_name = st.sidebar.text_input(
            f"要素名称 {i+1}", 
            value=f"factor_{i+1}", 
            key=f"factor_name_{i}"
        )
        emission_factor = st.sidebar.number_input(
            f"排放因子 {i+1} (kg CO2/单位)", 
            value=1.0, 
            step=0.1, 
            key=f"emission_factor_{i}"
        )
        correlation = st.sidebar.selectbox(
            f"与碳排放相关性 {i+1}", 
            ["positive", "negative"], 
            format_func=lambda x: "正相关 (数值越大，排放越大)" if x == "positive" else "负相关 (数值越大，排放越小)", 
            key=f"correlation_{i}"
        )
        
        custom_factors[factor_name] = {
            'emission_factor': emission_factor,
            'correlation': correlation
        }
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
                # 为小样本生成更有差异的数据
                base_value = 100.0
                if correlation_settings[col] == 'positive':
                    row[col + corr_text] = base_value + i * 50.0  # 创造更大差异
                else:
                    row[col + corr_text] = base_value - i * 30.0  # 负相关数据
            default_data.append(row)
        df_manual = pd.DataFrame(default_data)
    
    edited_df = st.data_editor(df_manual, num_rows="dynamic", use_container_width=True)
    
    # 列名映射
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
    
    # 构建结果表格
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
    
    # 权重信息表格
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
    
    # 检查权重差异并给出说明
    weight_std = np.std(weights)
    weight_range = np.max(weights) - np.min(weights)
    
    if weight_std < 0.05:
        st.warning(f"⚠️ 权重差异较小 (标准差: {weight_std:.4f})，这可能是由于样本数量少或指标间相关性较高导致的")
    else:
        st.success(f"✅ 权重分布合理 (标准差: {weight_std:.4f}，权重范围: {weight_range:.4f})")
    
    # 可视化分析
    st.subheader("📈 可视化分析")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['process_name'], df['carbon_emission'], 
                     color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df))))
        ax.set_title('各流程碳排放对比', fontproperties=font_prop)
        ax.set_ylabel('碳排放量 (kg CO2)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        
        # 设置x轴标签字体
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
            
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, 
                   f'{height:.1f}', ha='center', va='bottom', fontproperties=font_prop)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        
        # 创建带相关性标注的标签
        labels_with_correlation = []
        for col in used_columns:
            correlation_type = "+"
            if correlation_settings and col in correlation_settings:
                correlation_type = "+" if correlation_settings[col] == "positive" else "-"
            labels_with_correlation.append(f"{col}({correlation_type})")
        
        wedges, texts, autotexts = ax.pie(weights, labels=labels_with_correlation, 
                                        autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('熵权法 - 各指标权重分布\n(+正相关, -负相关)', fontproperties=font_prop)
        
        # 设置饼图标签字体
        for text in texts:
            text.set_fontproperties(font_prop)
        for autotext in autotexts:
            autotext.set_fontproperties(font_prop)
        
        st.pyplot(fig)
    
    # 其他分析图表
    if 'emission_intensity' in df.columns:
        st.subheader("📊 排放强度分析")
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(df['process_name'], df['emission_intensity'], 
                     color=plt.cm.Blues(np.linspace(0.3, 0.8, len(df))))
        ax.set_title('各流程碳排放强度对比', fontproperties=font_prop)
        ax.set_ylabel('排放强度 (kg CO2/单位产品)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
            
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, 
                   f'{height:.3f}', ha='center', va='bottom', fontproperties=font_prop)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 综合评分分析
    st.subheader("🎯 综合评分分析")
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = df.sort_values('weighted_score', ascending=True)
    bars = ax.barh(sorted_df['process_name'], sorted_df['weighted_score'], 
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df))))
    ax.set_title('各流程综合评分对比 (考虑权重和相关性)', fontproperties=font_prop)
    ax.set_xlabel('加权综合评分', fontproperties=font_prop)
    
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_prop)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_prop)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2., 
               f'{width:.3f}', ha='left' if width >= 0 else 'right', 
               va='center', fontproperties=font_prop)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 要素相关性分析（仅自定义模式）
    if custom_factors and correlation_settings:
        st.subheader("🔍 要素相关性分析")
        factor_columns = [col for col in used_columns if col in df.columns]
        if len(factor_columns) > 1:
            corr_matrix = df[factor_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f', ax=ax)
            ax.set_title('评估要素相关性矩阵', fontproperties=font_prop)
            
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(font_prop)
            for tick in ax.get_yticklabels():
                tick.set_fontproperties(font_prop)
            
            st.pyplot(fig)
        
        # 要素影响力分析
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
        
        # 影响力可视化
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if corr == '正相关' else 'red' for corr in influence_df['相关性']]
        bars = ax.bar(influence_df['要素名称'], influence_df['影响力得分'], 
                     color=colors, alpha=0.7)
        ax.set_title('各要素影响力得分 (绿色:正相关, 红色:负相关)', fontproperties=font_prop)
        ax.set_ylabel('影响力得分 (权重 × 标准差)', fontproperties=font_prop)
        ax.tick_params(axis='x', rotation=45)
        
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
            
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, 
                   f'{height:.3f}', ha='center', va='bottom', fontproperties=font_prop)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 改进建议
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
    
    # 具体改进建议
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
        st.subheader("📈 敏感性分析（弹性系数）")
    sensitivity_data = []
    
    # 计算均值
    emission_mean = df['carbon_emission'].mean()
    
    for col in used_columns:
        if col in df.columns and col != 'carbon_emission':
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            if col_mean > 0 and col_std > 0:
                # 弹性系数 = (排放量变化百分比) / (因素变化百分比)
                # 用变异系数近似
                emission_cv = df['carbon_emission'].std() / emission_mean
                factor_cv = col_std / col_mean
                
                if factor_cv > 0:
                    elasticity = emission_cv / factor_cv
                    
                    sensitivity_data.append({
                        '要素': col,
                        '弹性系数': elasticity,
                        '因素变异系数': factor_cv,
                        '排放量变异系数': emission_cv
                    })
    
    if sensitivity_data:
        sensitivity_df = pd.DataFrame(sensitivity_data)
        sensitivity_df = sensitivity_df.sort_values('弹性系数', ascending=False)
        st.write("**弹性系数排序** (表示因素变化1%时排放量变化百分之几):")
        st.dataframe(sensitivity_df.round(4))
    
    # 在 display_results 函数中，找到 "结果导出" 部分，替换为以下代码：

    st.subheader("📥 结果导出")
    col1, col2 = st.columns(2)
    
    with col1:
        # 准备详细分析结果数据
        export_df = df.copy()
        export_df['评估模式'] = "自定义模式" if custom_factors else "预设模式"
        for i, col in enumerate(used_columns):
            if col in export_df.columns:
                export_df[f'{col}_权重'] = weights[i]
        result_csv = export_df.to_csv(index=False).encode('utf-8-sig')
        
        # 使用 st.download_button 替代 st.button
        st.download_button(
            label="📊 下载详细分析结果",
            data=result_csv,
            file_name=f"碳排放评估详细结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_results"  # 添加唯一key
        )
    
    with col2:
        # 准备权重配置数据
        config_data = []
        for i, col in enumerate(used_columns):
            config_data.append({
                '要素名称': col,
                '权重': weights[i],
                '相关性': correlation_settings.get(col, '正相关') if correlation_settings else '正相关',
                '排放因子': custom_factors.get(col, {}).get('emission_factor', '系统默认') if custom_factors else '系统默认'
            })
        config_df = pd.DataFrame(config_data)
        config_csv = config_df.to_csv(index=False).encode('utf-8-sig')
        
        # 使用 st.download_button 替代 st.button
        st.download_button(
            label="📋 下载权重配置",
            data=config_csv,
            file_name=f"评估权重配置_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_config"  # 添加唯一key
        )

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
        electricity_factor = st.sidebar.number_input("电力排放因子 (kg CO2/kWh)", value=0.6205, step=0.01)
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
        # 使用 session_state 存储中间变量，避免 rerun 后丢失
        if 'analyzed' not in st.session_state:
            st.session_state['analyzed'] = False

        # 开始分析按钮 —— 点击后把结果写入 session_state
        if st.button("🚀 开始分析", key="start_analysis"):
            if assessment_mode == "自定义模式 (用户定义要素)":
                analyzed_df, weights, details_list, used_columns = assessment.assess_multiple_processes(
                    df, custom_columns, correlation_settings)
            else:
                analyzed_df, weights, details_list, used_columns = assessment.assess_multiple_processes(df)

            if analyzed_df is not None:
                # 将结果持久化到 session_state（可序列化的对象）
                st.session_state['analyzed'] = True
                st.session_state['analyzed_df'] = analyzed_df
                st.session_state['analyzed_weights'] = weights
                st.session_state['analyzed_details'] = details_list
                st.session_state['analyzed_used_columns'] = used_columns
                # 保存当前模式与配置，便于 display 和下载使用
                st.session_state['assessment_mode'] = assessment_mode
                st.session_state['custom_factors'] = custom_factors if assessment_mode == "自定义模式 (用户定义要素)" else None
                st.session_state['correlation_settings'] = correlation_settings if assessment_mode == "自定义模式 (用户定义要素)" else None

        # 如果 session_state 表示此前已经分析过（或刚刚分析完），使用持久化结果渲染界面
        if st.session_state.get('analyzed', False):
            # 从 session_state 恢复
            analyzed_df = st.session_state['analyzed_df']
            weights = st.session_state['analyzed_weights']
            details_list = st.session_state['analyzed_details']
            used_columns = st.session_state['analyzed_used_columns']
            display_results(
                analyzed_df, weights, details_list, used_columns,
                assessment,
                st.session_state.get('custom_factors', None),
                st.session_state.get('correlation_settings', None)
            )
    
  
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




