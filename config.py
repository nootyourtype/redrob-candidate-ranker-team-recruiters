import re

# 1. Strict Pure Services Penalty List (Lowercase for matching)
SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "larsen & toubro", "mphasis", "hexaware"
}

# 2. Core JD Requirements for Heuristic Scoring
MUST_HAVE_CONCEPTS = {
    "retrieval", "ranking", "search", "recommendation", "matching", "vector", "semantic", "query",
    "machine learning", "ml", "llm", "embedding", "rag"
}
VECTOR_DBS = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vespa"}
CORE_LANG = {"python"}
EVALUATION_TERMS = {"ndcg", "mrr", "map", "offline evaluation", "online evaluation", "ab test", "ab testing", "evaluation framework", "metrics", "experiment"}
HYBRID_SEARCH_TERMS = {"hybrid search", "semantic search", "dense retrieval", "sparse retrieval", "bm25", "vector search", "fusion search"}

BAD_DOMAINS = {
    "computer vision", "cv", "speech", "robotics", "autonomous", "self-driving", "image classification", "gesture", "vision"
}
RESEARCH_SIGNAL_TERMS = {
    "research", "paper", "publication", "iclr", "neurips", "arxiv", "researcher", "academic", "thesis", "state of the art", "novel", "propose"
}
CLOSED_SYSTEM_TERMS = {
    "proprietary", "closed system", "in-house", "legacy system", "black box", "confidential", "internal platform", "enterprise only"
}
FICTIONAL_COMPANIES = {
    "dunder mifflin", "stark industries", "wayne enterprises", "acme corp", "hooli", "pied piper", "initech", "globex inc", "umbrella corp", "oscorp"
}

PREFERRED_CITIES = {"pune", "noida", "gurgaon", "gurugram", "delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"}
PRIMARY_CITY_PREFERENCE = {"pune", "noida"}
SECONDARY_CITY_PREFERENCE = {"delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"}

JD_TEXT = """
Senior AI Engineer building ranking, retrieval, and matching systems.
Needs production experience owning retrieval and ranking systems, embeddings-based search, hybrid search, and vector database deployment.
Must be comfortable solving real user problems and shipping product-grade systems under time constraints.
Candidate should have strong Python fluency, evaluation framework experience, and a product orientation over research.
"""

# 4. Multipliers & Thresholds
GHOST_RESPONSE_RATE_THRESHOLD = 0.25
GHOST_INACTIVE_DAYS_THRESHOLD = 60
NOTICE_PERIOD_PREFERRED = 60
IDEAL_YOE_MIN = 5.0
IDEAL_YOE_MAX = 9.0
IDEAL_AI_YOE = 5.0
MAX_NOTICE_DAYS = 90
