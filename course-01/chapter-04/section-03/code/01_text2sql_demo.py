import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class Config:
    api_key_env: str = "API_KEY"
    db_name: str = "text2sql_demo.db"
    text2sql_dir: Path = Path(__file__).resolve().parent / "text2sql"


sys.path.append(str(Config().text2sql_dir))
from text2sql.text2sql_agent import SimpleText2SQLAgent  # noqa: E402


# -----------------------------------------------------------------------------
# 1.  创建演示数据库
# -----------------------------------------------------------------------------
def create_demo_database(cfg: Config) -> Path:
    db_path = Path.cwd() / cfg.db_name
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER,
                city TEXT
            )
            """
        )

        # 创建产品表
        cursor.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price REAL,
                stock INTEGER
            )
            """
        )

        # 创建订单表
        cursor.execute(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                order_date TEXT,
                total_price REAL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
            """
        )

        # 插入示例数据
        users_data = [
            (1, "张三", "zhangsan@email.com", 25, "北京"),
            (2, "李四", "lisi@email.com", 32, "上海"),
            (3, "王五", "wangwu@email.com", 28, "广州"),
            (4, "赵六", "zhaoliu@email.com", 35, "深圳"),
            (5, "陈七", "chenqi@email.com", 29, "杭州"),
        ]

        products_data = [
            (1, "iPhone 15", "电子产品", 7999.0, 50),
            (2, "MacBook Pro", "电子产品", 12999.0, 20),
            (3, "Nike运动鞋", "服装", 599.0, 100),
            (4, "办公椅", "家具", 899.0, 30),
            (5, "台灯", "家具", 199.0, 80),
            (6, "iPad", "电子产品", 3999.0, 40),
            (7, "Adidas外套", "服装", 399.0, 60),
        ]

        orders_data = [
            (1, 1, 1, 1, "2024-01-15", 7999.0),
            (2, 2, 3, 2, "2024-01-16", 1198.0),
            (3, 3, 5, 1, "2024-01-17", 199.0),
            (4, 1, 2, 1, "2024-01-18", 12999.0),
            (5, 4, 4, 1, "2024-01-19", 899.0),
            (6, 5, 6, 1, "2024-01-20", 3999.0),
            (7, 2, 7, 1, "2024-01-21", 399.0),
        ]

        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users_data)
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders_data)

    logging.info("演示数据库已创建: %s", db_path)
    return db_path


# -----------------------------------------------------------------------------
# 2.  设置演示环境
# -----------------------------------------------------------------------------
def setup_demo(cfg: Config) -> Optional[Tuple[SimpleText2SQLAgent, Path]]:
    logging.info("=== Text2SQL框架演示 ===")
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        logging.error("先设置%s环境变量", cfg.api_key_env)
        return None

    logging.info("创建演示数据库...")
    db_path = create_demo_database(cfg)

    logging.info("初始化Text2SQL代理...")
    agent = SimpleText2SQLAgent(api_key=api_key)

    logging.info("连接数据库...")
    if not agent.connect_database(str(db_path)):
        logging.error("数据库连接失败!")
        return None

    logging.info("加载知识库...")
    try:
        agent.load_knowledge_base()
    except Exception as exc:  # noqa: BLE001
        logging.error("知识库加载失败: %s", exc)
        return None

    logging.info("知识库加载成功!")
    return agent, db_path


# -----------------------------------------------------------------------------
# 3.  运行演示查询
# -----------------------------------------------------------------------------
def run_demo_queries(agent: SimpleText2SQLAgent) -> None:
    demo_questions = [
        "查询所有用户的姓名和邮箱",
        "年龄大于30的用户有哪些",
        "哪些产品的库存少于50",
        "查询来自北京的用户的所有订单",
        "统计每个城市的用户数量",
        "查询价格在500-8000之间的产品",
    ]

    logging.info("开始运行演示查询...")
    for idx, question in enumerate(demo_questions, start=1):
        logging.info("问题 %s: %s", idx, question)
        try:
            result = agent.query(question)
            if result["success"]:
                logging.info("成功! SQL: %s", result["sql"])
                results = result["results"]
                if isinstance(results, dict) and "rows" in results:
                    count = results["count"]
                    logging.info("返回 %s 行数据", count)
                    if count > 0:
                        preview_rows = results["rows"][:2]
                        for row_idx, row in enumerate(preview_rows, start=1):
                            row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
                            logging.info("  %s. %s", row_idx, row_str)
                        if count > 2:
                            logging.info("  ... 还有 %s 行", count - 2)
                else:
                    logging.info("结果: %s", results)
            else:
                logging.error("失败: %s", result["error"])
                logging.error("SQL: %s", result["sql"])
        except Exception as exc:  # noqa: BLE001
            logging.exception("执行错误: %s", exc)


# -----------------------------------------------------------------------------
# 4.  清理资源
# -----------------------------------------------------------------------------
def cleanup(agent: Optional[SimpleText2SQLAgent], db_path: Path) -> None:
    logging.info("清理资源...")
    if agent:
        agent.cleanup()
    if db_path.exists():
        db_path.unlink()
        logging.info("已删除演示数据库: %s", db_path)


def main() -> None:
    cfg = Config()
    setup_result = setup_demo(cfg)
    if setup_result is None:
        return

    agent, db_path = setup_result
    try:
        run_demo_queries(agent)
    finally:
        cleanup(agent, db_path)


if __name__ == "__main__":
    main()