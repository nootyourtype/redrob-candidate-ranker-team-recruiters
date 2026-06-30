# config.py - Configuration constants and patterns for V3 candidate ranking.
from datetime import datetime

REFERENCE_DATE = datetime(2026, 6, 12)
CUTOFF_PRE_LLM = datetime(2022, 11, 1).date()

TIER1_LOCATIONS = {"pune", "noida"}
TIER2_LOCATIONS = {"hyderabad", "delhi", "ncr", "gurugram", "gurgaon",
                   "mumbai", "bangalore", "bengaluru", "chennai",
                   "kolkata", "kochi", "jaipur", "ahmedabad"}

SALARY_BAND_MIN = 25.0
SALARY_BAND_MAX = 80.0

SHIPPED_RETRIEVAL = [
    "retrieval system","ranking system","recommendation system","search system",
    "search engine","ranking pipeline","retrieval pipeline","recommendation engine",
    "search ranking","ad ranking","ads ranking","product ranking","feed ranking",
    "reranking","re-ranking","ranking model","learning to rank","information retrieval",
    "query understanding","search relevance","candidate retrieval","document retrieval",
    "semantic search","hybrid search","vector search","dense retrieval","sparse retrieval",
    "faiss","elasticsearch","opensearch","solr","lucene","pinecone","weaviate","qdrant",
    "milvus","chroma","bm25","ann index","approximate nearest","embedding index",
    "served to users","serving millions","real users","production search",
    "personalization system","content ranking","item ranking","result ranking",
]
PRE_LLM_RETRIEVAL = [
    "information retrieval","learning to rank","bm25","lucene","solr","elasticsearch",
    "search relevance","query understanding","collaborative filtering","matrix factorization",
    "implicit feedback","click-through rate","ctr prediction","item2vec","word2vec",
    "ranknet","lambdamart","listwise","pairwise","xgboost","lightgbm",
]
PRODUCTION_SIGNALS = [
    "deployed to production","shipped to","launched","in production","serving",
    "real users","millions of","billion","at scale","latency","throughput","qps",
    "p99","sla","a/b test","a/b experiment","online evaluation","online eval",
    "production system","production traffic","rollout","canary","monitoring",
    "alerting","on-call","owned end-to-end",
]
CODING_SIGNALS = [
    "implemented","built","developed","wrote","coded","engineered","designed",
    "fine-tuned","fine tuned","shipped","deployed","optimised","optimized",
    "refactored","debugged","maintained","drove","migrated","owned","trained",
    "benchmarked","prototyped","contributed to","open-sourced",
]
PYTHON_SIGNALS = [
    "python","pytorch","tensorflow","numpy","pandas","sklearn","scikit-learn",
    "fastapi","flask","pydantic","asyncio","celery","sqlalchemy","pyspark",
]
EVAL_SIGNALS = [
    "ndcg","mrr","map@","precision@","recall@","hit rate","evaluation framework",
    "offline eval","online eval","a/b test","interleaving","counterfactual","uplift",
    "ranking metric","retrieval metric","relevance judgment",
]
SYSTEM_SIGNALS = [
    "distributed","large-scale","high-throughput","low-latency","scalable",
    "fault-tolerant","microservice","stream processing","kafka","spark","flink",
    "kinesis","event-driven","cache","sharding","replication",
]
AI_TERMS = [
    "machine learning","deep learning","neural network","nlp",
    "natural language processing","retrieval","ranking","recommendation",
    "embedding","pytorch","tensorflow","sklearn","model training",
    "model serving","feature engineering","a/b testing",
]
VIDEO_PRIMARY = [
    "video recommendation","video ranking","content recommendation","watch history",
    "streaming","video embeddings","video search","video retrieval",
    "youtube recommendation","watch time",
]
CV_SPEECH_ROBOTICS = [
    "computer vision","object detection","image classification","speech recognition",
    "speech synthesis","text-to-speech","automatic speech","asr","tts","robotics",
    "autonomous driving","lidar","point cloud","3d detection","pose estimation",
]
CONSULTING_FIRMS = [
    "tcs","infosys","wipro","accenture","cognizant","capgemini","hcl",
    "tech mahindra","mphasis","hexaware","mindtree","ltimindtree",
]
CONSULTING_INDUSTRIES = [
    "it services","consulting","outsourcing","bpo",
    "information technology and services","staffing",
]
JD_VECTOR_DB = [
    "faiss","pinecone","weaviate","qdrant","milvus","chroma","elasticsearch",
    "opensearch","pgvector","vespa","typesense","hybrid search","vector search",
    "semantic search","dense retrieval","sparse retrieval","ann index","approximate nearest",
]
JD_RETRIEVAL_EVAL = [
    "retrieval","ranking","recommendation","learning to rank","reranking","re-ranking",
    "bm25","ndcg","mrr","map@","precision@","recall@","information retrieval",
    "ranking model","search relevance","query understanding",
]
JD_LLM_FT = [
    "llm","fine-tun","fine tun","lora","qlora","peft","rlhf",
    "rag","retrieval augmented","instruction tuning","sft",
]
JD_HR_DOMAIN = [
    "hr tech","recruiting","talent","marketplace","job board",
    "applicant tracking","candidate matching","hiring",
]
CULTURE_NEGATIVE = [
    "stable codebase","mature codebase","clear specs","predictable",
    "well-documented codebase","stable environment","low ambiguity",
    "clear requirements","need clear","dislike ambiguity","need stable",
    "structured environment","clear specifications","well-defined",
    "need documentation","requires stability",
]
CULTURE_POSITIVE = [
    "ship fast","move fast","iterate","pragmatic","ship and iterate",
    "comfortable with ambiguity","scrappy","fast-paced","quick experiments",
    "bias for action","break assumptions","ship it","get things done","wear many hats",
]
NON_COMPETE_TERMS = [
    "non-compete","non compete","restrictive covenant",
    "non-solicitation","garden leave","cooling off period",
]
MANAGER_TITLES = [
    "vp ","vice president","chief ","cto","ceo","coo","head of","director",
    "senior manager","general manager","programme director","group manager",
]
SENIOR_IC_TITLES = [
    "staff engineer","staff scientist","principal engineer",
    "distinguished engineer","senior staff","tech lead",
]
FICTIONAL_COMPANIES = {
    "dunder mifflin","stark industries","wayne enterprises","acme corp",
    "hooli","pied piper","initech","globex inc","umbrella corp","oscorp",
}

DIMENSION_WEIGHTS = {
    "jd_alignment": 0.30,
    "technical":    0.30,
    "production":   0.25,
    "availability": 0.08,
    "behavior":     0.07,
}

UNREACH_CAP           = 0.65
VISA_CAP              = 0.62
INTEGRITY_CAP         = 0.50
GHOST_INACTIVE_DAYS   = 120
GHOST_RRR_THRESHOLD   = 0.10
UNREACH_INACTIVE_DAYS = 90
UNREACH_RRR_THRESHOLD = 0.20
IDEAL_YOE_MIN = 5.0
IDEAL_YOE_MAX = 9.0
