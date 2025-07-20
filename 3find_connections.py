import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Local imports
from database_handler import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConnectionFinder:
    def __init__(self, model_name: str = "meta-llama/llama-4-maverick"):
        """
        Initialize the ConnectionFinder with a specific model.
        
        Args:
            model_name: Name of the model to use via OpenRouter
        """
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.chain = self._create_chain()
        # Chain for matching phrase to unit
        self.match_chain = self._create_match_chain()
        # DB handler
        self.db = DatabaseHandler()
    
    def _initialize_llm(self):
        """Initialize the language model with OpenRouter."""
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            max_retries=3,
            temperature=0.2,
            max_tokens=1000,
        )
    
    def _create_chain(self):
        """Создаем цепочку для анализа иерархических зависимостей."""
        template = """Ты эксперт по анализу учебных материалов. 
        Определи, какие знания необходимы для понимания темы "{title}".
        Учитывай только темы, которые действительно являются обязательными предпосылками.
        
        Информация о юните:
        Тип: {kind}
        Название: {title}
        Утверждение: {statement}
        Теги: {tags}
        
        Верни JSON-массив с названиями тем в порядке их изучения (от базовых к сложным).
        Пример: ["Базовое понятие", "Более сложная тема", "Целевая тема"]
        
        Текущая тема: {title}
        
        Сгенерируй только JSON-массив, без дополнительного текста."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return {
            "kind": lambda x: x.get("kind", ""),
            "title": lambda x: x.get("title", ""),
            "statement": lambda x: x.get("statement", ""),
            "proof": lambda x: x.get("proof", ""),
            "tags": lambda x: ", ".join(x.get("tags", []))
        } | prompt | self.llm | StrOutputParser()

    def _create_match_chain(self):
        """Создаем цепочку для проверки, является ли тема предпосылкой."""
        template = """Ты эксперт по анализу учебных зависимостей.
        Определи, является ли знание темы "{prereq}" необходимой предпосылкой 
        для понимания темы "{title}".

        Критерии:
        1. Тема является обязательной для понимания
        2. Без неё невозможно полноценно изучить целевую тему

        Ответь одним словом: 'YES' если да, 'NO' если нет.

        Пример 1:
        Пререквизит: Сложение
        Тема: Умножение
        Ответ: YES

        Пример 2:
        Пререквизит: Интегралы
        Тема: Производные
        Ответ: NO

        Пререквизит: {prereq}
        Тема: {title}
        Утверждение: {statement}

        Ответ (только YES или NO):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {
                "prereq": lambda x: x["phrase"],
                "title": lambda x: x["unit"].get("title", ""),
                "statement": lambda x: x["unit"].get("statement", ""),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _is_prerequisite(self, prereq: str, unit: dict) -> bool:
        """Проверяет, является ли тема prereq необходимой предпосылкой для unit."""
        if not prereq or not prereq.strip():
            return False
            
        try:
            # Проверяем, что prereq не является частью самого юнита
            if prereq.lower() in unit.get("title", "").lower() \
               or prereq.lower() in unit.get("statement", "").lower():
                return False
                
            # Проверяем через LLM
            response = self.match_chain.invoke({
                "phrase": prereq,
                "unit": {
                    "title": unit.get("title", ""),
                    "statement": unit.get("statement", "")
                }
            }).strip().upper()
            
            return response.startswith("YES")
            
        except Exception as e:
            logger.error(f"Error checking prerequisite: {str(e)}")
            return False
    
    def find_connections(self, unit: Dict[str, Any]) -> List[str]:
        """
        Find potential connections for a given knowledge unit.
        
        Args:
            unit: Dictionary containing unit information (kind, title, statement, proof, tags)
            
        Returns:
            List of strings representing potential connections
        """
        try:
            # Get the raw response from the LLM
            response = self.chain.invoke(unit)
            
            # Clean up the response to extract JSON array
            response = response.strip()
            if response.startswith('```json'):
                response = response[response.find('['):response.rfind(']')+1]
            elif '```' in response:
                response = response[response.find('['):response.rfind('`')].strip()
            
            # Parse the JSON response
            connections = json.loads(response)
            
            if not isinstance(connections, list):
                logger.warning(f"Unexpected response format: {response}")
                return []
                
            # Clean and validate the connections
            connections = [
                str(conn).strip() 
                for conn in connections 
                if conn and str(conn).strip()
            ]
            
            logger.info(f"Found {len(connections)} potential connections for unit: {unit.get('title', 'Untitled')}")
            return connections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}\nResponse: {response}")
            return []
        except Exception as e:
            logger.error(f"Error finding connections: {str(e)}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # High-level orchestration: iterate over all units and discover links
    # ------------------------------------------------------------------
    def _has_dependency_cycle(self, source_id: str, target_id: str) -> bool:
        """Проверяет, не создаст ли связь цикл в графе зависимостей."""
        visited = set()
        to_visit = [target_id]
        
        while to_visit:
            current = to_visit.pop()
            if current == source_id:
                return True
            if current not in visited:
                visited.add(current)
                # Получаем все зависимости current
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "SELECT target_id FROM connections WHERE source_id = ?", 
                    (str(current),)
                )
                to_visit.extend(row[0] for row in cursor.fetchall())
        return False

    def discover_all_connections(self, top_k: int = 10, save: bool = False):
        """Walk through every knowledge unit and create connections using LLM verification.

        Args:
            top_k: количество кандидатов для проверки на фразу
            save: сохранять ли связи в БД
            
        Returns:
            List[Tuple[int,int]]: список найденных связей (source_id, target_id)
        """
        discovered = []
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM knowledge_units")
        all_units = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Running discovery for {len(all_units)} units …")

        for target_unit in all_units:
            target_id = target_unit["id"]
            
            # Получаем список потенциальных пререквизитов
            prereqs = self.find_connections(target_unit)
            if not prereqs:
                continue
                
            logger.info(f"Found {len(prereqs)} potential prerequisites for: {target_unit.get('title', 'Untitled')}")
            
            for prereq in prereqs:
                # Ищем юниты, соответствующие пререквизиту
                candidates = self.db.search_units_by_text(prereq, limit=top_k)
                
                for candidate in candidates:
                    source_id = candidate["id"]
                    
                    # Пропускаем петли и существующие связи
                    if (source_id == target_id or 
                        (source_id, target_id) in discovered or
                        self._has_dependency_cycle(source_id, target_id)):
                        continue
                    
                    # Проверяем, действительно ли это пререквизит
                    if self._is_prerequisite(candidate.get('title', ''), target_unit):
                        discovered.append((source_id, target_id))
                        
                        if save:
                            try:
                                self.db.add_connection(
                                    str(source_id), 
                                    str(target_id), 
                                    relationship_type="prerequisite"
                                )
                                logger.debug(f"Saved connection: {source_id} → {target_id}")
                            except Exception as e:
                                logger.error(f"Failed to save connection ({source_id}→{target_id}): {e}")

        logger.info(f"Discovered {len(discovered)} connections via LLM verification")
        return discovered


def configure_logging():
    """Настройка логирования в файл и консоль."""
    # Создаем директорию для логов, если её нет
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Формат логов
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Базовый конфиг
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # Логи уровня INFO и выше в файл
            logging.FileHandler(f"logs/connections_{timestamp}.log"),
            # Логи уровня WARNING и выше в консоль
            logging.StreamHandler(),
        ],
    )
    
    # Уровень логирования для httpx (библиотека запросов)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def main():
    """
    Основная функция для запуска поиска связей.
    
    Запуск:
        python 3find_connections.py  # Только поиск
        python 3find_connections.py --save  # Поиск и сохранение в БД
    """
    import argparse
    
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Поиск связей между учебными материалами')
    parser.add_argument('--no-save', action='store_false', dest='save', help='Не сохранять найденные связи в БД')
    parser.add_argument('--top-k', type=int, default=10, help='Количество кандидатов для проверки на фразу')
    args = parser.parse_args()
    
    # Настройка логирования
    configure_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"Запуск поиска связей (save={args.save}, top_k={args.top_k})")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # Инициализация и запуск поиска
        finder = ConnectionFinder()
        logger.info("Инициализация ConnectionFinder завершена")
        
        # Запуск поиска связей
        logger.info("Начинаем поиск связей...")
        connections = finder.discover_all_connections(top_k=args.top_k, save=args.save)
        
        # Вывод статистики
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Поиск завершен за {elapsed:.1f} сек")
        logger.info(f"Найдено связей: {len(connections)}")
        
        if connections:
            logger.info("\nПримеры найденных связей:")
            for src, tgt in connections[:5]:  # Показываем первые 5 связей
                logger.info(f"  {src} → {tgt}")
                
                # Сохраняем связь в БД, если не сохраняли в discover_all_connections
                if args.save:
                    try:
                        finder.db.add_connection(str(src), str(tgt), relationship_type="llm_related")
                        logger.debug(f"Сохранена связь: {src} → {tgt}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении связи {src} → {tgt}: {str(e)}")
            
            if len(connections) > 5:
                logger.info(f"  ... и ещё {len(connections) - 5} связей")
        
        logger.info("=" * 60)
        logger.info("Поиск связей завершён. Все связи сохранены в базе данных.")
        
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nПрервано пользователем")
    except Exception as e:
        logger.exception("Ошибка при поиске связей:")
    finally:
        logger.info("Работа завершена")


if __name__ == "__main__":
    import time
    from datetime import datetime
    from pathlib import Path
    
    main()