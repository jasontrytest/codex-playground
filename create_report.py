import json
from datetime import datetime

import openai
import config


client = openai.OpenAI(api_key=config.OPENAI_API_KEY)


def get_market_insights():
    """Call OpenAI API to fetch latest market insights and return as dict."""
    instruction = (
        """核心指令一：洞察抓取指令
作为一名专业的宏观经济数据分析助手，请抓取并整理今日全球市场的关键信息。\n"
        "请以JSON格式返回，包含以下字段：\n"
        "- global_markets: 全球主要股票市场指数的涨跌情况;\n"
        "- commodities: 重要大宗商品价格及变化;\n"
        "- forex: 主要货币对的走势;\n"
        "- economic_events: 当日需要关注的重大经济事件或数据发布。\n"
        """
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data assistant."},
                {"role": "user", "content": instruction},
            ],
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("解析JSON失败：API返回的数据不是有效的JSON。")
            return None
    except Exception as err:
        print(f"调用OpenAI接口失败：{err}")
        return None


def compose_report(insights_data):
    """Generate macroeconomic report from insights_data."""
    instruction_template = (
        """核心指令二：报告生成指令
请根据以下JSON数据撰写一份结构完整、语言专业的每日宏观经济报告：\n
[此处将插入由第一个指令生成的JSON数据]\n
要求：\n1. 先概述整体市场情绪；\n2. 分模块分析股票市场、大宗商品、外汇等；\n3. 结合经济事件给出后市展望。\n"""
    )
    filled_instruction = instruction_template.replace(
        "[此处将插入由第一个指令生成的JSON数据]",
        json.dumps(insights_data, ensure_ascii=False, indent=2),
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一名资深的宏观经济分析师。"},
                {"role": "user", "content": filled_instruction},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as err:
        print(f"生成报告时发生错误：{err}")
        return None


if __name__ == "__main__":
    print("---[ 开始生成每日宏观经济报告 ]---")
    print("[1/3] 正在获取市场最新洞察...")
    data = get_market_insights()
    if data is None:
        print("获取市场洞察失败，程序终止。")
        raise SystemExit(1)
    print("[2/3] 洞察已获取，正在撰写分析报告...")
    report = compose_report(data)
    if report is None:
        print("生成报告失败，程序终止。")
        raise SystemExit(1)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Macro_Report_{today}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[3/3] 报告生成完毕！已保存至文件: {filename}")
