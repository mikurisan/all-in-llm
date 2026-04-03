from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from collections import Counter

import logging
import hashlib
import uuid

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """Data ETL

    attrs:
        data_path:
        documents: parent documents (full recipe)
        chunks: child documents (splitted by title)
        parent_child_map: ID mapping relationships between parent and child document
    """

    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        """
        args:
            data_path: data folder path
        """
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
    
    def load_documents(self) -> List[Document]:
        """Load markdown documents from `self.data_path`

        returns:
            loaded document list
        """

        # get absolute path object
        data_root = Path(self.data_path).resolve()
        logger.info("Loading documents from %s ...", data_root)

        documents: List[Document] = []

        for md_file in data_root.rglob("*.md"):
            md_file = md_file.resolve()
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read file %s: %s", md_file, e)
                continue
            # use relative paths helps create stable parent id
            try:
                relative_path = md_file.relative_to(data_root).as_posix()
            except ValueError:
                relative_path = md_file.as_posix()

            parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "relative_path": relative_path,
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )

            documents.append(doc)

        # enhance metadata, failure doesn't disrupt the overall process
        for doc in documents:
            try:
                self._enhance_metadata(doc)
            except Exception as e:
                logger.warning(
                    "Failed to enhance metadata for %s: %s",
                    doc.metadata.get("source", "<unknown>"),
                    e,
                )

        self.documents = documents
        logger.info("Successfully loaded %d documents from %s", len(documents), data_root)
        return documents
    
    def _enhance_metadata(self, doc: Document) -> None:
        """Enhance document metadata.

        Populate the following metadata fields based on the
        document's source path:

        - `category`: dish category
        - `dish_name`: dish name (filename without the extension)
        - `difficulty`: difficulty level
        
        args:
            document requiring metadata enhancement
        """

        metadata: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "")

        file_path = Path(source) if source else Path()
        path_parts = file_path.parts
        
        metadata["category"] = self._get_category_from_path(path_parts)
        metadata["dish_name"] = file_path.stem or "未知菜品"
        content: str = getattr(doc, "page_content", "") or ""
        metadata["difficulty"] = self._get_difficulty_from_content(content)

        doc.metadata = metadata
        
    def _get_category_from_path(self, path_parts: tuple[str, ...]) -> str:
        """
        args:
            path_parts: splitted file path using path separator
                e.g., "/dir/sub/xx.md" -> ["/", "dir", "sub", "xx.md"]
        """
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                return value
        return "其他"

    def _get_difficulty_from_content(self, content: str) -> str:
        """
        args:
            content: markdown file content
        """
        difficulty_mapping = {
            "★★★★★": "非常困难",
            "★★★★": "困难",
            "★★★": "中等",
            "★★": "简单",
            "★": "非常简单",
        }
        for stars, difficulty in difficulty_mapping.items():
            if stars in content:
                return difficulty
        return "未知"

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique identifier for a chunk."""
        return str(uuid.uuid4())

    def chunk_documents(self) -> List[Document]:
        """Split loaded documents into Markdown-structure-aware chunks

        returns:
            a list of chunked documents with basic metadata
        """
        logger.info("Chunking documents ...")

        if not self.documents:
            raise ValueError("Load the document first.")

        chunks = self._markdown_header_split()

        # add basic metadata to each chunk
        for index, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_id", self._generate_chunk_id())
            chunk.metadata['batch_index'] = index
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info("Successfully chunked a total of %d chunks",len(chunks))
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """Split content into structured sections using `MarkdownHeaderTextSplitter`

        returns:
            a document list splitted by header
        """
        # specify the heading levels to split
        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题")
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            # keep heading for context comprehension
            strip_headers=False
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # check if the document content includes heading
                content_preview = doc.page_content[:200]
                has_headers = any(
                    line.lstrip().startswith("#")
                    for line in content_preview.splitlines()
                )
                if not has_headers:
                    logger.warning("No header found in document content for dish %s", doc.metadata.get('dish_name', '未知'))
                    logger.debug("Content preview: %s", content_preview)

                md_chunks = markdown_splitter.split_text(doc.page_content)

                parent_id = doc.metadata["parent_id"]

                for index, chunk in enumerate(md_chunks):
                    child_id = self._generate_chunk_id()

                    new_metadata = {
                        **doc.metadata,
                        **chunk.metadata,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": index,
                    }
                    chunk.metadata = new_metadata

                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning("Document %s splitted failed.", e)
                all_chunks.append(doc)

        logger.info("Successfully splitted %d chunks", len(all_chunks))

        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """Filter documents by category"""
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """Filter documents by difficulty
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics of loaded documents and chunks"""
        if not self.documents:
            return {}

        category_counter = Counter()
        difficulty_counter = Counter()

        for doc in self.documents:
            metadata = getattr(doc, "metadata", {}) or {}
            category = metadata.get("category", "未知")
            difficulty = metadata.get("difficulty", "未知")

            category_counter[category] += 1
            difficulty_counter[difficulty] += 1

        total_chunks = len(self.chunks)
        if total_chunks:
            total_chunk_size = sum(
                chunk.metadata.get("chunk_size", 0)
                for chunk in self.chunks
            )
            avg_chunk_size = total_chunk_size / total_chunks
        else:
            avg_chunk_size = 0

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': dict(category_counter),
            'difficulties': dict(difficulty_counter),
            'avg_chunk_size': avg_chunk_size
        }

    def export_metadata(self, output_path: str):
        """Export documents's metadata to a json file"""
        import json

        path = Path(output_path)
        metadata_list = [
            {
                "source": doc.metadata.get("source"),
                "dish_name": doc.metadata.get("dish_name"),
                "category": doc.metadata.get("category"),
                "difficulty": doc.metadata.get("difficulty"),
                "content_length": len(getattr(doc, "page_content", "") or ""),
            }
            for doc in self.documents
        ]

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info("Successfully exported metadata to: %s", path)

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """Get the corresponding parent document by child chunk (smart deduplication)

        args:
            child_chunks: retrieved child chunks

        returns:
            an ordered list of parent documents
        """

        parent_docs_index: Dict[Any, Document] = {}
        for doc in self.documents:
            parent_id = doc.metadata.get("parent_id")
            if parent_id is not None:
                parent_docs_index[parent_id] = doc

        # count the number of matches for each parent document (relevance metric)
        parent_relevance = Counter()
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id is not None:
                parent_relevance[parent_id] += 1

        # sorted by relevance
        sorted_parent_ids = [
            parent_id
            for parent_id, _ in parent_relevance.most_common()
            if parent_id in parent_docs_index
        ]

        # a deduplicated list of parent documents
        parent_docs: List[Document] = [parent_docs_index[parent_id] for parent_id in sorted_parent_ids]

        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(
            "%d deduplicated parent documents found from %d chunks: %s",
            len(child_chunks),
            len(parent_docs),
            ", ".join(parent_info),
        )

        return parent_docs