from neo4j import GraphDatabase

# --- 配置 ---
URI = "bolt://localhost:7687"
AUTH = ("测试", "aa20020619")  # 填你的密码


# -----------

def clean_silent():
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                # 1. 删数据
                session.run("MATCH (n) DETACH DELETE n")

                # 2. 删约束
                for rec in session.run("SHOW CONSTRAINTS"):
                    session.run(f"DROP CONSTRAINT {rec['name']}")

                # 3. 删索引 (跳过系统索引)
                for rec in session.run("SHOW INDEXES"):
                    if rec["type"] != "system" and "LOOKUP" not in rec["type"]:
                        session.run(f"DROP INDEX {rec['name']}")

        print("✅ 数据库已清空。")
    except Exception as e:
        print(f"❌ 出错: {e}")


if __name__ == "__main__":
    clean_silent()