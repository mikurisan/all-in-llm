import sys
import logging
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

    def initialize_system(self):
        logger.info("🚀 Initializing RAG system...")

        logger.info("Initializing data preparation module...")
        self.data_module = DataPreparationModule(self.config.data_path)

        logger.info("Initializing index construction module...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        logger.info("🤖 Initializing generation integration module...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        logger.info("✅ System initialization completed!")
    
    def build_knowledge_base(self):
        logger.info("Constructing knowledge base...")

        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            logger.info("✅ Successfully loaded saved vector index!")
            logger.info("Loading recipe documents...")
            self.data_module.load_documents()
            logger.info("Performing text chunking...")
            chunks = self.data_module.chunk_documents()
        else:
            logger.info("No saved index found, building new index...")

            logger.info("Loading recipe documents...")
            self.data_module.load_documents()

            logger.info("Performing text chunking...")
            chunks = self.data_module.chunk_documents()

            logger.info("Building vector index...")
            vectorstore = self.index_module.build_vector_index(chunks)

            logger.info("Saving vector index...")
            self.index_module.save_index()

        logger.info("Initializing retrieval optimization module...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        logger.info(f"📊 Knowledge base statistics:")
        logger.info(f"   Total documents: {stats['total_documents']}")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Categories: {list(stats['categories'].keys())}")
        logger.info(f"   Difficulty distribution: {stats['difficulties']}")

        logger.info("✅ Knowledge base construction completed!")

    def ask_question(self, question: str, stream: bool = False):

        if not (self.retrieval_module and self.generation_module):
            raise ValueError("Construct the knowledge base before asking questions.")

        logger.info("❓ User question: %s", question)

        route_type = self.generation_module.query_router(question)
        logger.info("🎯 Query type: %s", route_type)

        if route_type == "list":
            rewritten_query = question
            logger.info("📝 List query kept as is: %s", question)
        else:
            logger.info("🤖 Intelligent analysis of query...")
            rewritten_query = self.generation_module.query_rewrite(question)

        logger.info("🔍 Retrieving relevant documents...")
        filters = self._extract_filters_from_query(question)
        kwargs = {"top_k": self.config.top_k}
        if filters:
            logger.info(" Applying filter conditions: %s", filters)
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, **kwargs
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, **kwargs
            )

        def _section_title(preview: str) -> str:
            if preview.startswith("#"):
                title_end = preview.find("\n")
                title_end = len(preview) if title_end == -1 else title_end
                return preview[:title_end].replace("#", "").strip()
            return "内容片段"

        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                preview = chunk.page_content[:100].strip()
                chunk_info.append(f"{dish_name}({_section_title(preview)})")
            logger.info("Found %d relevant documents: %s", len(relevant_chunks), ', '.join(chunk_info))
        else:
            logger.info("Found 0 relevant documents.")
            return "Sorry, couldn't find relevant recipe info. Try other dish name or keywords."

        if route_type == "list":
            logger.info("📋 Generating dish list...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            doc_names = [
                doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs
            ]
            if doc_names:
                logger.info("Found documents: %s", ', '.join(doc_names))
            return self.generation_module.generate_list_answer(question, relevant_docs)

        logger.info("Getting complete documents...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = [
            doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs
        ]
        if doc_names:
            logger.info("Found documents: %s", ', '.join(doc_names))
        else:
            logger.info("Found %d complete documents", len(relevant_docs))

        logger.info("✍️ Generating detailed answer...")

        if route_type == "detail":
            generator = (
                self.generation_module.generate_step_by_step_answer_stream
                if stream
                else self.generation_module.generate_step_by_step_answer
            )
        else:
            generator = (
                self.generation_module.generate_basic_answer_stream
                if stream
                else self.generation_module.generate_basic_answer
            )
        return generator(question, relevant_docs)

    
    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}

        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """Search dishes by category with optional query filtering"""

        if not self.retrieval_module:
            raise ValueError("Construct the knowledge base before asking questions.")
        
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """Get the list of ingredients for a specific dish"""

        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("Construct the knowledge base before asking questions.")

        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        answer = self.generation_module.generate_basic_answer(f"{dish_name} 需要什么食材？", docs)

        return answer
    
    def run_interactive(self):
        self.initialize_system()
        
        self.build_knowledge_base()
        
        print("Interavtive RAG system is ready! You can ask questions about recipes. Type 'exit' or 'quit' to exit.")
        
        while True:
            try:
                user_input = input("Question: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                stream_choice = input("Use stream output? (y/n, default y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("Answer:")
                if use_stream:
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error occurred while processing question: {e}")
        
def main():
    try:
        rag_system = RecipeRAGSystem()
        
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error("System error: %s", e)

if __name__ == "__main__":
    main()
