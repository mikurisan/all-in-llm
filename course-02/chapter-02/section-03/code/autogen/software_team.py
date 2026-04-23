import os
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from textwrap import dedent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

def create_openai_model_client():
    return OpenAIChatCompletionClient(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

def create_product_manager(model_client):
    system_message = dedent("""
        As an experienced software product manager.

        Core responsibilities include:
        1. **Requirements Analysis**: Deeply understand user needs, identify core functions and boundary conditions
        2. **Technical Planning**: Develop a clear technical implementation path based on the requirements
        3. **Risk Assessment**: Identify potential technical risks and user experience issues
        4. **Coordination and Communication**: Communicate effectively with engineers and other team members

        When receiving a development task, please analyze it according to the following structure:
        1. Requirement understanding and analysis
        2. Functional module division
        3. Technical selection recommendations
        4. Implementation priority ordering
        5. Acceptance criteria definition

        Respond concisely and clearly.
                            
        Add "Engineers begin implementation" at the end of response.
    """)

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )

def create_engineer(model_client):
    system_message = dedent("""
        As an experienced Python web development engineer.
                            
        Technical expertise includes:
        1. **Python Programming**: Proficient in Python syntax and best practices
        2. **Web Development**: Skilled in frameworks such as Streamlit, Flask, Django
        3. **API Integration**: Extensive experience in third-party API integration
        4. **Error Handling**: Focus on code robustness and exception handling

        When receiving a development task, please:
        1. Carefully analyze the technical requirements
        2. Select appropriate technical solutions
        3. Write complete code implementations
        4. Add necessary comments and documentation
        5. Consider edge cases and exception handling

        Provide complete runnable code.

        Add "Code reviewers begin review" at the end of response.
    """)

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )

def create_code_reviewer(model_client):
    system_message = dedent("""
        As an experienced code reviewer.

        Review focus includes:
        1. **Code Quality**: Check the readability, maintainability, and performance of the code
        2. **Security**: Identify potential security vulnerabilities and risk points
        3. **Best Practices**: Ensure the code follows industry standards and best practices
        4. **Error Handling**: Verify the completeness and合理性 of exception handling

        Review Process:
        1. Carefully read and understand the code logic
        2. Check code conventions and best practices
        3. Identify potential issues and improvement points
        4. Provide specific modification suggestions
        5. Evaluate the overall quality of the code

        Provide specific review comments.
                            
        Add "Code review completed, user proxy start testing" at the end of response.
    """)

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )

def create_user_proxy():
    return UserProxyAgent(
        name="UserProxy",
        description=dedent("""
            As a user proxy, responsible for:
            1.  Represent users to propose development requirements
            2.  Execute the final code implementation
            3.  Verify whether the functionality meets expectations
            4.  Provide user feedback and suggestions
            After completing the testing, please reply with TERMINATE.
        """),
    )

async def run_software_development_team():
    
    logger.info("🔧 Initializing model client...")
    
    model_client = create_openai_model_client()
    
    logger.info("👥 Creating intelligent agent team...")
    
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()
    
    termination = TextMentionTermination("TERMINATE")
    
    team_chat = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer, 
            code_reviewer,
            user_proxy
        ],
        termination_condition=termination,
        max_turns=20,
    )
    
    task = dedent("""
        我们需要开发一个比特币价格显示应用，具体要求如下：

        核心功能：
        - 实时显示比特币当前价格（USD）
        - 显示24小时价格变化趋势（涨跌幅和涨跌额）
        - 提供价格刷新功能

        技术要求：
        - 使用 Streamlit 框架创建 Web 应用
        - 界面简洁美观，用户友好
        - 添加适当的错误处理和加载状态

        请团队协作完成这个任务，从需求分析到最终实现。
        """
    )

    logger.info("🚀 Starting AutoGen software development team collaboration...")
    
    result = await Console(team_chat.run_stream(task=task))
    
    logger.info("✅ Collaboration completed.")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(run_software_development_team())

    print(f"\nFinal Result: {result}")