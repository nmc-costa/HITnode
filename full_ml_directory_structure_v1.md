

---

# 📁 ML Pipeline Directory Structure

This codebase follows a node-based architecture where each directory contains reusable nodes that can be composed into ML pipelines.

## 📁 Directory Structure

```
# Phase 1: Business & Data Understanding (CRISP-ML(Q) + AI Integration)
business_understanding/      # Business objectives and stakeholder requirements processing
├── requirements/           # NLP nodes to extract and embed stakeholder requirements from documents
├── constraints/            # Nodes to parse technical/business constraints and create constraint graphs
├── risk_assessment/        # Automated risk analysis nodes using NLP and rule-based systems
├── success_metrics/        # Nodes to extract and formalize business KPIs and success criteria
├── stakeholder_analysis/   # NLP nodes to analyze stakeholder communications and priorities
├── domain_knowledge/       # Nodes to process domain-specific documents and create knowledge graphs
└── context_extraction/     # Extract and structure business context for LLM applications

data_understanding/          # Data exploration and automated analysis
├── exploratory_analysis/   # Automated EDA nodes with statistical profiling
├── data_quality/          # Data quality assessment and anomaly detection nodes
├── statistics/            # Statistical analysis and distribution modeling nodes
├── hypothesis_generation/ # NLP nodes to generate data-driven hypotheses from initial analysis
├── semantic_analysis/     # Nodes to understand data semantics and relationships
├── metadata_extraction/   # Automated metadata extraction and cataloging nodes
└── multimodal_analysis/   # Analysis nodes for text, image, audio, and video data

# Phase 2: Data Preparation (CRISP-DM/ML(Q) + LLM Data Processing)
datasets/                    # Data I/O, ingestion, loading, and saving
├── loaders/                # Load data from various sources (CSV, JSON, APIs, databases)
├── savers/                 # Save data to different formats and destinations
├── extractors/             # Extract data from external systems and APIs
├── validators/             # Data quality validation and schema checking
├── huggingface_datasets/   # Hugging Face dataset integration and management
└── multimodal_loaders/     # Loaders for text, image, audio, video data

preprocessing/               # Data cleaning, transformations, feature engineering
├── cleaners/               # Data cleaning and quality improvement nodes
├── transformers/           # Data type conversions and transformations
├── feature_engineering/    # Feature creation, selection, and extraction
├── normalizers/            # Data normalization and scaling
├── splitters/              # Train/validation/test data splitting
├── text_processors/        # Text preprocessing for NLP and LLM applications
├── tokenizers/             # Tokenization nodes for various models and frameworks
└── embeddings_prep/        # Prepare data for embedding generation

# Phase 3: Model Engineering (CRISP-ML(Q) + LLM/AI Models)
models/                      # Training, fitting, prediction, and inference
├── trainers/               # Model training and fitting nodes
├── predictors/             # Prediction and inference nodes
├── tuners/                 # Hyperparameter tuning and optimization
├── architectures/          # Model architecture definitions and configurations
├── ensembles/             # Ensemble methods and model combination
├── versioning/            # Model versioning and comparison
├── huggingface_models/    # Hugging Face model integration and fine-tuning
├── llm_models/            # Large language model implementations and wrappers
├── embedding_models/      # Embedding model implementations (text, image, multimodal)
├── multimodal_models/     # Multi-modal AI models (vision-language, etc.)
└── custom_architectures/  # Custom neural network architectures

# LLM/AI Application Components
llm_applications/            # LLM-powered application components
├── prompts/               # Prompt engineering and template management
│   ├── templates/         # Reusable prompt templates
│   ├── chains/            # Prompt chaining strategies
│   ├── few_shot/          # Few-shot learning prompts
│   └── optimization/      # Prompt optimization and testing
├── agents/                # AI agent implementations and workflows
│   ├── autonomous/        # Autonomous agent systems
│   ├── collaborative/     # Multi-agent collaboration
│   ├── tool_using/        # Tool-using agents
│   └── planning/          # Agent planning and reasoning
├── chains/                # LangChain-style processing chains
│   ├── sequential/        # Sequential processing chains
│   ├── parallel/          # Parallel processing chains
│   ├── conditional/       # Conditional logic chains
│   └── custom/            # Custom chain implementations
├── tools/                 # AI tools and function calling
│   ├── api_tools/         # API interaction tools
│   ├── data_tools/        # Data manipulation tools
│   ├── search_tools/      # Search and retrieval tools
│   └── custom_tools/      # Custom tool implementations
├── retrievers/            # Information retrieval systems
│   ├── vector_search/     # Vector-based retrieval
│   ├── keyword_search/    # Traditional keyword search
│   ├── hybrid_search/     # Hybrid retrieval approaches
│   └── semantic_search/   # Semantic search implementations
├── rag_systems/           # Retrieval Augmented Generation
│   ├── retrievers/        # RAG retrieval components
│   ├── generators/        # RAG generation components
│   ├── rankers/           # Result ranking and reranking
│   ├── fusion/            # Multi-source information fusion
│   └── evaluation/        # RAG system evaluation
├── context_management/    # Context handling and memory
│   ├── memory/            # Conversation and session memory
│   ├── context_windows/   # Context window management
│   ├── summarization/     # Context summarization strategies
│   └── retrieval/         # Context retrieval mechanisms
└── mcp_integration/       # Model Context Protocol integration
    ├── servers/           # MCP server implementations
    ├── clients/           # MCP client implementations
    ├── connectors/        # MCP connectors for various tools
    └── protocols/         # Custom protocol implementations

# Phase 4: Model Evaluation (CRISP-ML(Q) + LLM Evaluation)
evaluation/                  # Unified evaluation, metrics, and analysis
├── metrics/                # Performance metrics (MAE, MSE, F1, AUC, NDCG, BLEU, ROUGE)
│   ├──regression/             # Regression metrics (MAE, MSE, RMSE, MAPE, R²)
│   ├── classification/         # Classification metrics (accuracy, F1, precision, recall, AUC)
│   ├── ranking/                # Ranking metrics (NDCG, MAP, MRR)
│   ├── custom/                 # Business-specific and domain metrics
scoring/                     # Scoring predictions and business logic
│   ├──thresholding/           # Threshold-based scoring and classification
│   ├── ranking/                # Ranking and percentile-based scoring
│   ├── calibration/            # Probability calibration and confidence scoring
│   ├── business_rules/         # Business logic and rule-based scoring
├── validation/             # Cross-validation and model validation strategies
├── testing/                # A/B testing and statistical testing nodes
├── explainability/         # Model interpretability (SHAP, LIME, feature importance)
├── comparison/             # Model comparison and benchmarking
├── quality_assurance/      # Quality gates and acceptance criteria
├── stakeholder_review/     # Stakeholder acceptance and sign-off
├── llm_evaluation/         # LLM-specific evaluation methods
│   ├── human_eval/        # Human evaluation frameworks
│   ├── automated_eval/    # Automated evaluation systems
│   ├── bias_testing/      # Bias and fairness evaluation
│   ├── safety_testing/    # AI safety and alignment testing
│   └── prompt_evaluation/ # Prompt effectiveness evaluation
└── rag_evaluation/         # RAG system evaluation
    ├── retrieval_metrics/ # Retrieval quality metrics
    ├── generation_metrics/# Generation quality metrics
    ├── end_to_end/        # End-to-end RAG evaluation
    └── human_feedback/    # Human feedback integration

# Phase 5: Model Deployment (CRISP-ML(Q) + LLM Deployment)
deployment/                  # Production deployment and serving
├── serving/                # Real-time and batch serving endpoints
├── infrastructure/         # Infrastructure as code and deployment configs
├── pipelines/             # Automated deployment pipelines
├── rollback/              # Rollback and recovery mechanisms
├── security/               # Security configurations and access controls
├── llm_serving/           # LLM-specific serving infrastructure
│   ├── api_endpoints/     # LLM API endpoint implementations
│   ├── streaming/         # Streaming response handling
│   ├── batching/          # Batch processing systems
│   └── load_balancing/    # Load balancing for LLM services
├── agent_deployment/      # AI agent deployment systems
│   ├── orchestration/     # Agent orchestration platforms
│   ├── workflow_engines/  # Workflow execution engines
│   ├── scheduling/        # Agent task scheduling
│   └── monitoring/        # Agent performance monitoring
└── edge_deployment/       # Edge and mobile deployment
    ├── optimization/      # Model optimization for edge
    ├── quantization/      # Model quantization techniques
    ├── compression/       # Model compression methods
    └── mobile_integration/# Mobile app integration

# Phase 6: Monitoring & Maintenance (CRISP-ML(Q) + LLM Operations)
monitoring/                  # Continuous monitoring and maintenance
├── performance/            # Model performance and drift monitoring
├── data_quality/          # Ongoing data quality monitoring
├── alerts/                # Alerting and notification systems
├── maintenance/           # Model maintenance and retraining triggers
├── feedback_loops/        # Feedback collection and incorporation
├── lifecycle_management/   # Model lifecycle and retirement
├── llm_monitoring/        # LLM-specific monitoring
│   ├── token_usage/       # Token consumption tracking
│   ├── latency_tracking/  # Response time monitoring
│   ├── quality_monitoring/# Output quality assessment
│   ├── safety_monitoring/ # Safety and alignment monitoring
│   └── cost_optimization/ # Cost tracking and optimization
├── agent_monitoring/      # AI agent monitoring systems
│   ├── task_tracking/     # Agent task execution tracking
│   ├── goal_achievement/  # Goal completion monitoring
│   ├── error_handling/    # Agent error detection and recovery
│   └── learning_metrics/  # Agent learning and improvement tracking
└── user_feedback/         # User interaction and satisfaction monitoring
    ├── satisfaction/      # User satisfaction tracking
    ├── usage_patterns/    # Usage pattern analysis
    ├── feature_requests/  # Feature request management
    └── issue_tracking/    # User issue and bug tracking

# Supporting Infrastructure
conf/                        # Configuration management and environment settings
├── base/                   # Base configuration files
├── local/                  # Local development overrides
├── environments/           # Environment-specific configurations
└── quality_gates/          # Quality assurance configurations

storage/                     # All storage operations (RAG, vector DBs, model storage, LLM data)
├── vector_databases/       # Vector database operations for embeddings and semantic search
│   ├── pinecone/          # Pinecone integration nodes
│   ├── weaviate/          # Weaviate integration nodes
│   ├── chroma/            # Chroma DB integration nodes
│   ├── faiss/             # FAISS integration nodes
│   └── custom/            # Custom vector database implementations
├── model_registry/         # Model versioning, storage, and retrieval
│   ├── huggingface_hub/   # Hugging Face model hub integration
│   ├── mlflow_registry/   # MLflow model registry
│   ├── local_registry/    # Local model storage
│   └── cloud_registry/    # Cloud-based model storage
├── embeddings/             # Store and retrieve embeddings for RAG and semantic applications
│   ├── text_embeddings/   # Text embedding storage and retrieval
│   ├── image_embeddings/  # Image embedding storage and retrieval
│   ├── multimodal_embeddings/ # Multi-modal embedding storage
│   └── custom_embeddings/ # Custom embedding implementations
├── feature_store/          # Feature store management and serving
├── knowledge_graphs/       # Graph databases for constraints, relationships, and domain knowledge
│   ├── neo4j/             # Neo4j graph database integration
│   ├── rdf_stores/        # RDF triple store integration
│   └── property_graphs/   # Property graph implementations
├── document_stores/        # Document storage for RAG and knowledge systems
│   ├── elasticsearch/     # Elasticsearch document storage
│   ├── mongodb/           # MongoDB document storage
│   ├── postgres/          # PostgreSQL with document features
│   └── custom_stores/     # Custom document storage implementations
├── conversation_memory/    # Conversation and session memory storage
│   ├── short_term/        # Short-term memory implementations
│   ├── long_term/         # Long-term memory implementations
│   ├── semantic_memory/   # Semantic memory systems
│   └── episodic_memory/   # Episodic memory implementations
└── cache/                  # Caching mechanisms for performance optimization
    ├── llm_cache/         # LLM response caching
    ├── embedding_cache/   # Embedding computation caching
    ├── retrieval_cache/   # Retrieval result caching
    └── general_cache/     # General purpose caching

governance/                  # Compliance, audit, and policy management
├── policies/               # Data governance and usage policies
├── audit_trails/           # Audit logs and compliance tracking
├── privacy/                # Privacy-preserving techniques and GDPR compliance
├── risk_management/        # Risk assessment and mitigation strategies
├── stakeholder_management/ # Stakeholder communication and approval workflows
└── regulatory_compliance/  # Regulatory compliance (EU AI Act, FDA, etc.)

tests/                       # Comprehensive testing framework
├── unit/                   # Unit tests for individual nodes
├── integration/            # Integration tests for pipelines
├── data_validation/        # Data quality and schema validation tests
├── model_validation/       # Model performance and quality tests
└── acceptance/             # User acceptance and stakeholder tests

docs/                        # Documentation and compliance records
├── api/                    # API documentation
├── tutorials/              # Usage examples and tutorials
├── compliance/             # Regulatory compliance documentation
├── architecture/           # System architecture documentation
├── methodology/           # CRISP-ML(Q) process documentation
└── stakeholder_reports/   # Reports for business stakeholders

utils/                       # Common utilities and helper functions
├── data_helpers/           # Data manipulation and processing utilities
├── model_helpers/          # Model-related utility functions
├── io_helpers/             # Input/output operation helpers
├── security_helpers/       # Security and encryption utilities
├── nlp_helpers/            # NLP utilities for text processing and analysis
├── graph_helpers/          # Graph database and knowledge graph utilities
├── crisp_helpers/          # CRISP-DM/ML(Q) workflow utilities and phase transition managers
├── llm_helpers/            # LLM integration and management utilities
│   ├── api_clients/       # LLM API client utilities (OpenAI, Anthropic, etc.)
│   ├── prompt_utils/      # Prompt engineering and optimization utilities
│   ├── token_management/  # Token counting and management utilities
│   ├── response_parsing/  # LLM response parsing and validation
│   └── cost_tracking/     # LLM usage cost tracking utilities
├── langchain_helpers/      # LangChain integration and workflow utilities
│   ├── chain_builders/    # Chain construction utilities
│   ├── agent_utils/       # Agent management utilities
│   ├── tool_integrations/ # Tool integration helpers
│   └── memory_utils/      # Memory and context management utilities
├── huggingface_helpers/    # Hugging Face integration utilities
│   ├── model_loaders/     # Model loading and management utilities
│   ├── dataset_utils/     # Dataset processing utilities
│   ├── tokenizer_utils/   # Tokenizer management utilities
│   └── pipeline_utils/    # Pipeline construction utilities
├── rag_helpers/            # RAG system utilities and optimizations
│   ├── retrieval_utils/   # Retrieval optimization utilities
│   ├── chunking_utils/    # Document chunking and processing utilities
│   ├── embedding_utils/   # Embedding generation and management utilities
│   └── fusion_utils/      # Multi-source information fusion utilities
├── mcp_helpers/            # Model Context Protocol utilities
│   ├── server_utils/      # MCP server development utilities
│   ├── client_utils/      # MCP client integration utilities
│   ├── protocol_utils/    # Protocol implementation utilities
│   └── discovery_utils/   # Tool discovery and registration utilities
├── multimodal_helpers/     # Multi-modal AI utilities
│   ├── vision_utils/      # Computer vision utilities
│   ├── audio_utils/       # Audio processing utilities
│   ├── text_vision_utils/ # Text-vision integration utilities
│   └── modality_fusion/   # Cross-modal fusion utilities
└── agent_helpers/          # AI agent development utilities
    ├── planning_utils/    # Agent planning and reasoning utilities
    ├── execution_utils/   # Agent execution and control utilities
    ├── collaboration_utils/ # Multi-agent collaboration utilities
    └── learning_utils/    # Agent learning and adaptation utilities
```

## 🔄 Node-Based Architecture

Each directory contains **nodes** - reusable components that can be combined to build ML pipelines. Nodes follow consistent interfaces for easy composition and testing.

## 📈 Key Improvements from Industry Research & CRISP-ML(Q) + LLM/AI Alignment

- **CRISP-ML(Q) Structure**: Organized directories by CRISP-ML(Q) phases for systematic ML development
- **LLM/AI Integration**: Comprehensive support for modern AI applications including RAG, agents, and multi-modal systems
- **Business Understanding Integration**: Added dedicated `business_understanding/` and stakeholder management capabilities
- **Data Understanding Phase**: Dedicated `data_understanding/` with multi-modal analysis capabilities
- **LLM Application Framework**: Complete `llm_applications/` directory supporting prompts, agents, chains, tools, RAG, and MCP integration
- **Modern AI Patterns**: Support for LangChain, Hugging Face, RAG systems, AI agents, and Model Context Protocol
- **Quality Assurance Framework**: Integrated quality gates and stakeholder review processes throughout
- **Configuration Management**: Added `conf/` directory for centralized configuration following Kedro/MLflow patterns
- **Unified Evaluation**: Comprehensive evaluation framework with LLM-specific evaluation and stakeholder acceptance criteria
- **Security & Governance**: Enhanced with AI safety, responsible AI practices, and comprehensive compliance requirements
- **Testing Infrastructure**: Multi-layer testing framework supporting CRISP-ML(Q) validation requirements plus LLM evaluation
- **Documentation & Compliance**: Structured documentation supporting regulatory requirements and stakeholder communication
- **Monitoring & Maintenance**: Dedicated phase for continuous monitoring with LLM-specific tracking and feedback loops
- **Risk Management**: Integrated AI risk assessment and mitigation throughout the ML lifecycle
- **Multi-modal Support**: Complete support for text, image, audio, and cross-modal AI applications

## 🎯 Design Principles (CRISP-ML(Q) + Modern AI Aligned)

- **Business-First**: Every component clearly supports business objectives and stakeholder requirements
- **AI-Native**: Built for modern LLM, RAG, agent, and multi-modal AI applications from the ground up
- **Iterative**: Support for CRISP-ML(Q)'s iterative and feedback-driven approach
- **Modularity**: Each node has a single responsibility within the CRISP-ML(Q) framework
- **Reusability**: Nodes can be used across different projects and CRISP-ML(Q) iterations
- **Scalability**: Easy to add new nodes and extend functionality within the methodology framework
- **Compliance-Ready**: Supports regulatory requirements and audit trails
- **Continuous Improvement**: Feedback loops and learning mechanisms embedded throughout
- **Framework-Agnostic**: Support for multiple AI frameworks (LangChain, Hugging Face, custom implementations)
- **Standards-Compliant**: Adherence to industry standards (MCP, scikit-learn interfaces, MLOps principles)
- **Performance-Optimized**: Built-in optimization for LLM costs, latency, and resource efficiency
- **Human-AI Collaboration**: Designed for seamless human-AI collaboration and oversight

