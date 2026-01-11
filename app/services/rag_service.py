"""Сервис RAG для семантического поиска релевантных фрагментов ТЗ."""
import os
from typing import List, Dict, Any, Optional
import structlog
from pathlib import Path

from app.core.config import settings
from app.utils.text_chunking import chunk_tz_text, TextChunk
from app.utils.file_handlers import load_examples

logger = structlog.get_logger(__name__)

# Глобальные переменные для lazy loading
_embedding_model = None
_chroma_client = None
_chroma_collection = None


def _get_embedding_model():
    """Lazy loading модели embeddings."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info('loading_embeddings_model', model=settings.RAG_EMBEDDINGS_MODEL)
            
            # Пробуем загрузить модель с fallback
            model_names = [
                settings.RAG_EMBEDDINGS_MODEL,
                "cointegrated/rubert-tiny2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ]
            
            for model_name in model_names:
                try:
                    _embedding_model = SentenceTransformer(model_name)
                    logger.info('embeddings_model_loaded', model=model_name)
                    break
                except Exception as e:
                    logger.warning('embeddings_model_load_failed', model=model_name, error=str(e))
                    continue
            
            if _embedding_model is None:
                raise RuntimeError("Не удалось загрузить ни одну модель embeddings")
                
        except ImportError:
            msg = "sentence-transformers не установлен. Установите: pip install sentence-transformers"
            logger.error('sentence_transformers_not_installed')
            raise ImportError(msg)
    
    return _embedding_model


def _get_chroma_client():
    """Lazy loading клиента Chroma."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            db_path = Path(settings.RAG_DB_PATH)
            db_path.mkdir(parents=True, exist_ok=True)
            
            _chroma_client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info('chroma_client_initialized', db_path=str(db_path))
            
        except ImportError:
            msg = "chromadb не установлен. Установите: pip install chromadb"
            logger.error('chromadb_not_installed')
            raise ImportError(msg)
    
    return _chroma_client


def _get_chroma_collection():
    """Получение или создание коллекции Chroma."""
    global _chroma_collection
    if _chroma_collection is None:
        client = _get_chroma_client()
        try:
            _chroma_collection = client.get_or_create_collection(
                name="tz_examples",
                metadata={"description": "Коллекция примеров ТЗ для RAG"}
            )
            logger.info('chroma_collection_loaded', collection_name="tz_examples")
        except Exception as e:
            logger.error('chroma_collection_error', error=str(e))
            raise
    
    return _chroma_collection


class RAGService:
    """Сервис для работы с RAG (Retrieval-Augmented Generation)."""
    
    def __init__(self):
        """Инициализация RAGService."""
        self.is_available = False
        self.embedding_model = None
        self.collection = None
        
        if not settings.RAG_ENABLED:
            logger.info('rag_disabled')
            return
        
        try:
            self.embedding_model = _get_embedding_model()
            self.collection = _get_chroma_collection()
            self.is_available = True
            logger.info('rag_service_initialized')
        except Exception as e:
            logger.warning('rag_service_init_failed', error=str(e))
            self.is_available = False
    
    def index_examples(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Индексация примеров ТЗ в векторную БД.

        Args:
            force_reindex: Принудительная переиндексация (даже если БД не пуста)

        Returns:
            Словарь с результатами индексации
        """
        if not self.is_available:
            logger.warning('rag_not_available_for_indexing')
            return {"status": "error", "message": "RAG недоступен"}
        
        try:
            collection = _get_chroma_collection()
            
            # Проверяем, нужно ли индексировать
            if not force_reindex and collection.count() > 0:
                logger.info('rag_already_indexed', count=collection.count())
                return {
                    "status": "skipped",
                    "message": "Индексация уже выполнена",
                    "chunks_count": collection.count()
                }
            
            # Загружаем примеры
            examples = load_examples(settings.EXAMPLES_DIR)
            if not examples:
                logger.warning('no_examples_to_index', examples_dir=settings.EXAMPLES_DIR)
                return {"status": "error", "message": "Нет примеров для индексации"}
            
            # Очищаем коллекцию при принудительной переиндексации
            if force_reindex:
                try:
                    client = _get_chroma_client()
                    client.delete_collection("tz_examples")
                    collection = _get_chroma_collection()
                    logger.info('rag_collection_cleared')
                except Exception as e:
                    logger.warning('rag_collection_clear_error', error=str(e))
            
            # Разбиваем на чанки и индексируем
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []
            
            chunk_counter = 0
            
            for example_idx, example_text in enumerate(examples):
                # Определяем источник (имя файла)
                source = f"example_{example_idx}"
                
                # Разбиваем на чанки
                chunks = chunk_tz_text(
                    text=example_text,
                    source=source,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    overlap=settings.RAG_CHUNK_OVERLAP,
                    language=settings.RAG_LANGUAGE,
                    chunk_by_sentences=settings.RAG_CHUNK_BY_SENTENCES
                )
                
                for chunk in chunks:
                    all_chunks.append(chunk.text)
                    all_metadatas.append({
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        "block_type": chunk.block_type
                    })
                    all_ids.append(f"{source}_chunk_{chunk.chunk_index}")
                    chunk_counter += 1
            
            # Генерируем embeddings батчами для оптимизации
            logger.info('generating_embeddings', chunks_count=len(all_chunks))
            embedding_model = _get_embedding_model()
            
            batch_size = 32
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings.tolist())
            
            # Добавляем в Chroma
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(
                'rag_indexing_complete',
                documents_count=len(examples),
                chunks_count=chunk_counter,
                total_chunks=collection.count()
            )
            
            return {
                "status": "success",
                "documents_count": len(examples),
                "chunks_count": chunk_counter,
                "total_chunks": collection.count()
            }
            
        except Exception as e:
            msg = f'Ошибка индексации: {e}'
            logger.error('rag_indexing_error', error=str(e))
            return {"status": "error", "message": msg}
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков по запросу.

        Args:
            query: Текст запроса (описание проекта)
            top_k: Количество релевантных чанков (по умолчанию из настроек)

        Returns:
            Список словарей с релевантными чанками и метаданными
        """
        if not self.is_available:
            logger.warning('rag_not_available_for_retrieval')
            return []
        
        if top_k is None:
            top_k = settings.RAG_TOP_K
        
        try:
            collection = _get_chroma_collection()
            embedding_model = _get_embedding_model()
            
            # Генерируем embedding для запроса
            query_embedding = embedding_model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0].tolist()
            
            # Поиск в векторной БД
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Формируем результат
            chunks = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    chunks.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else None,
                        "relevance_score": 1.0 - (results['distances'][0][i] if results['distances'] else 0.0)
                    })
            
            logger.debug(
                'rag_retrieval_complete',
                query_length=len(query),
                chunks_found=len(chunks)
            )
            
            return chunks
            
        except Exception as e:
            logger.error('rag_retrieval_error', error=str(e), query_preview=query[:50])
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики RAG.

        Returns:
            Словарь со статистикой
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "enabled": settings.RAG_ENABLED,
                "message": "RAG недоступен"
            }
        
        try:
            collection = _get_chroma_collection()
            count = collection.count()
            
            return {
                "status": "available",
                "enabled": settings.RAG_ENABLED,
                "chunks_count": count,
                "model": settings.RAG_EMBEDDINGS_MODEL,
                "language": settings.RAG_LANGUAGE
            }
        except Exception as e:
            logger.error('rag_stats_error', error=str(e))
            return {
                "status": "error",
                "enabled": settings.RAG_ENABLED,
                "message": str(e)
            }
