# 📁 ML Full Directory Structure

This codebase follows a node-based architecture where each directory contains reusable nodes that can be composed into ML pipelines, organized according to CRISP-ML(Q) methodology with modern AI/LLM integration.

## 📁 Directory Structure

```
# 📥 Phase 1: Business & Data Understanding 
business_understanding/          # 📋 Business: Business objectives and stakeholder requirements processing
├── requirements/               # 🎯 Business: NLP nodes to extract and embed stakeholder requirements from documents
├── constraints/                # ⚠️ Business: Nodes to parse technical/business constraints and create constraint graphs
├── risk_assessment/            # 🛡️ Business: Automated risk analysis nodes using NLP and rule-based systems
├── success_metrics/            # 📊 Business: Nodes to extract and formalize business KPIs and success criteria
├── stakeholder_analysis/       # 👥 Business: NLP nodes to analyze stakeholder communications and priorities
├── domain_knowledge/           # 📚 Business: Nodes to process domain-specific documents and create knowledge graphs
└── context_extraction/         # 🔍 Business: Extract and structure business context for LLM applications

data_understanding/              # 🔬 Analysis: Data exploration and automated analysis
├── exploratory_analysis/       # 📈 Analysis: Automated EDA nodes with statistical profiling
├── data_quality/               # ✅ Analysis: Data quality assessment and anomaly detection nodes (text, image, audio, video)
├── statistics/                 # 📊 Analysis: Statistical analysis and distribution modeling nodes
├── hypothesis_generation/      # 💡 Analysis: NLP nodes to generate data-driven hypotheses from initial analysis
├── semantic_analysis/          # 🧠 Analysis: Nodes to understand data semantics and relationships
├── metadata_extraction/        # 🏷️ Analysis: Automated metadata extraction and cataloging nodes
└── multimodal_analysis/        # 🎭 Analysis: Analysis nodes for text, image, audio, and video data

# 🛠 Phase 2: Data Engineering (Data Preparation) 
datasets/                        # 📥 Data I/O: ingestion, loading, and saving
├── loaders/                    # 📂 I/O: Load data from various sources (CSV, JSON, APIs, databases) and modalities
├── savers/                     # 💾 I/O: Save data to different formats and destinations
├── extractors/                 # 🔌 I/O: Extract data from external systems and APIs
├── validators/                 # ✔️ I/O: Data quality validation and schema checking
├── huggingface_datasets/       # 🤗 I/O: Hugging Face dataset integration and management
└── multimodal_loaders/         # 🎭 I/O: Loaders for text, image, audio, video data

preprocessing/                   # 🛠️ Data Transformations: Data cleaning, transformations, feature engineering
├── cleaners/                   # 🧹 Transform: Data cleaning and quality improvement nodes
├── transformers/               # 🔄 Transform: Data type conversions and transformations
├── feature_engineering/        # ⚙️ Transform: Feature creation, selection, and extraction
├── normalizers/                # 📏 Transform: Data normalization and scaling
├── splitters/                  # ✂️ Transform: Train/validation/test data splitting
├── text_processors/            # 📝 Transform: Text preprocessing for NLP and LLM applications
├── tokenizers/                 # 🔤 Transform: Tokenization nodes for various models and frameworks
├── chunking/                   # 📄 Transform: Chunking long text into smaller semantically meaningful "chunks"
└── embeddings_prep/            # 🎯 Transform: Prepare data for embedding generation

# 🧠 Phase 3: ML Model Engineering 
models/                          # 🤖 ML Core: Training, fitting, prediction, and inference
├── trainers/                   # 🎓 Training: Model training and fitting nodes
├── predictors/                 # 🔮 Training: Prediction and inference nodes
├── tuners/                     # 🎛️ Training: Hyperparameter tuning and optimization
├── architectures/              # 🏗️ Training: Model architecture definitions and configurations
├── ensembles/                  # 🤝 Training: Ensemble methods and model combination
├── registry/                   # 📦 Training: Documenting ML model, versioning, experiments metadata
├── huggingface_models/         # 🤗 Training: Hugging Face model integration and fine-tuning
├── llm_models/                 # 🧠 Training: Large language model implementations and wrappers
├── embedding_models/           # 🎯 Training: Embedding model implementations (text, image, multimodal)
├── multimodal_models/          # 🎭 Training: Multi-modal AI models (vision-language, etc.)
└── custom_architectures/       # 🔧 Training: Custom neural network architectures

gpai_applications/               # 🧠 GPAI: Large Models powered applications components (LLMOPs, VLMOPs, Transformers, agents)
├── prompts/                    # 💬 GPAI: Prompt engineering and template management
│   ├── templates/              # 📋 GPAI: Reusable prompt templates
│   ├── chains/                 # 🔗 GPAI: Prompt chaining strategies
│   ├── few_shot/               # 🎯 GPAI: Few-shot learning prompts
│   └── optimization/           # 🎛️ GPAI: Prompt optimization and testing
├── agents/                     # 🤖 GPAI: AI agent implementations and workflows
│   ├── autonomous/             # 🚀 GPAI: Autonomous agent systems
│   ├── collaborative/          # 🤝 GPAI: Multi-agent collaboration
│   ├── tool_using/             # 🛠️ GPAI: Tool-using agents
│   └── planning/               # 🎯 GPAI: Agent planning and reasoning
├── chains/                     # 🔗 GPAI: LangChain-style processing chains
│   ├── sequential/             # ➡️ GPAI: Sequential processing chains
│   ├── parallel/               # ⚡ GPAI: Parallel processing chains
│   ├── conditional/            # 🔄 GPAI: Conditional logic chains
│   └── custom/                 # 🔧 GPAI: Custom chain implementations
├── tools/                      # 🛠️ GPAI: AI tools and function calling
│   ├── api_tools/              # 🔌 GPAI: API interaction tools
│   ├── data_tools/             # 📊 GPAI: Data manipulation tools
│   ├── search_tools/           # 🔍 GPAI: Search and retrieval tools
│   └── custom_tools/           # 🔧 GPAI: Custom tool implementations
├── retrievers/                 # 🔍 GPAI: Information retrieval systems
│   ├── vector_search/          # 🎯 GPAI: Vector-based retrieval
│   ├── keyword_search/         # 🔤 GPAI: Traditional keyword search
│   ├── hybrid_search/          # 🔀 GPAI: Hybrid retrieval approaches
│   └── semantic_search/        # 🧠 GPAI: Semantic search implementations
├── rag_systems/                # 📚 GPAI: Retrieval Augmented Generation
│   ├── retrievers/             # 🔍 GPAI: RAG retrieval components
│   ├── generators/             # 🔮 GPAI: RAG generation components
│   ├── rankers/                # 🏆 GPAI: Result ranking and reranking
│   ├── fusion/                 # 🔀 GPAI: Multi-source information fusion
│   └── evaluation/             # 📊 GPAI: RAG system evaluation
├── context_management/         # 🧠 GPAI: Context handling and memory
│   ├── memory/                 # 💾 GPAI: Conversation and session memory
│   ├── context_windows/        # 🖼️ GPAI: Context window management
│   ├── summarization/          # 📝 GPAI: Context summarization strategies
│   └── retrieval/              # 🔍 GPAI: Context retrieval mechanisms
└── mcp_integration/            # 🔌 GPAI: Model Context Protocol integration
    ├── servers/                # 🖥️ GPAI: MCP server implementations
    ├── clients/                # 💻 GPAI: MCP client implementations
    ├── connectors/             # 🔗 GPAI: MCP connectors for various tools
    └── protocols/              # 📡 GPAI: Custom protocol implementations

# 🚀 Phase 4: ML Model Evaluation 
evaluation/                      # 📊 Assessment: Unified evaluation, metrics, and analysis
├── metrics/                    # 📈 Assessment: Performance metrics (MAE, MSE, F1, AUC, NDCG, BLEU, ROUGE)
│   ├── regression/             # 📊 Assessment: Regression metrics (MAE, MSE, RMSE, MAPE, R²)
│   ├── classification/         # 🎯 Assessment: Classification metrics (accuracy, F1, precision, recall, AUC)
│   ├── ranking/                # 🏆 Assessment: Ranking metrics (NDCG, MAP, MRR)
│   └── custom/                 # 🔧 Assessment: Business-specific and domain metrics
├── scoring/                    # 🎯 Assessment: Scoring predictions and business logic
│   ├── thresholding/           # 📏 Assessment: Threshold-based scoring and classification
│   ├── ranking/                # 🏆 Assessment: Ranking and percentile-based scoring
│   ├── calibration/            # 🎛️ Assessment: Probability calibration and confidence scoring
│   └── business_rules/         # 📋 Assessment: Business logic and rule-based scoring
├── validation/                 # ✅ Assessment: Cross-validation and model validation strategies
├── testing/                    # 🧪 Assessment: A/B testing and statistical testing nodes
├── explainability/             # 🔍 Assessment: Model interpretability (SHAP, LIME, feature importance)
├── comparison/                 # ⚖️ Assessment: Model comparison and benchmarking
├── quality_assurance/          # 🛡️ Assessment: Quality gates and acceptance criteria
├── stakeholder_review/         # 👥 Assessment: Stakeholder acceptance and sign-off
├── reports/                    # 📋 Assessment: Representations, plots, demos for different users
├── llm_evaluation/             # 🧠 Assessment: LLM-specific evaluation methods
│   ├── human_eval/             # 👥 Assessment: Human evaluation frameworks
│   ├── automated_eval/         # 🤖 Assessment: Automated evaluation systems
│   ├── bias_testing/           # ⚖️ Assessment: Bias and fairness evaluation
│   ├── safety_testing/         # 🛡️ Assessment: AI safety and alignment testing
│   └── prompt_evaluation/      # 💬 Assessment: Prompt effectiveness evaluation
└── rag_evaluation/             # 📚 Assessment: RAG system evaluation
    ├── retrieval_metrics/      # 🔍 Assessment: Retrieval quality metrics
    ├── generation_metrics/     # 🔮 Assessment: Generation quality metrics
    ├── end_to_end/             # 🔄 Assessment: End-to-end RAG evaluation
    └── human_feedback/         # 👥 Assessment: Human feedback integration

# 🏭 Phase 5: Model Deployment 
deployment/                      # 🚀 Production: Production deployment and serving
├── serving/                    # 🌐 Production: Real-time and batch serving endpoints, packaging, containers, Docker images
├── infrastructure/             # 🏗️ Production: Infrastructure as code and deployment configs
├── user_interfaces/            # 👥 Production: Developer, partner, end user interfaces, StreamLit labs
├── rollback/                   # ↩️ Production: Rollback and recovery mechanisms
├── security/                   # 🔒 Production: Security configurations and access controls
├── llm_serving/                # 🧠 Production: LLM-specific serving infrastructure
│   ├── api_endpoints/          # 🔌 Production: LLM API endpoint implementations
│   ├── streaming/              # 🌊 Production: Streaming response handling
│   ├── batching/               # 📦 Production: Batch processing systems
│   └── load_balancing/         # ⚖️ Production: Load balancing for LLM services
├── agent_deployment/           # 🤖 Production: AI agent deployment systems
│   ├── orchestration/          # 🎭 Production: Agent orchestration platforms
│   ├── workflow_engines/       # 🔄 Production: Workflow execution engines
│   ├── scheduling/             # 📅 Production: Agent task scheduling
│   └── monitoring/             # 👁️ Production: Agent performance monitoring
└── edge_deployment/            # 📱 Production: Edge and mobile deployment
    ├── optimization/           # 🎯 Production: Model optimization for edge
    ├── quantization/           # 📏 Production: Model quantization techniques
    ├── compression/            # 🗜️ Production: Model compression methods
    └── mobile_integration/     # 📱 Production: Mobile app integration

# 🔄 Phase 6: Monitoring & Maintenance 
monitoring/                      # 👁️ Operations: Continuous monitoring and maintenance
├── performance/                # 📊 Operations: Model performance and drift monitoring
├── data_quality/               # ✅ Operations: Ongoing data quality monitoring
├── alerts/                     # 🚨 Operations: Alerting and notification systems
├── maintenance/                # 🔧 Operations: Model maintenance and retraining triggers
├── feedback_loops/             # 🔄 Operations: Feedback collection and incorporation
├── lifecycle_management/       # 📊 Operations: Model lifecycle and retirement
├── llm_monitoring/             # 🧠 Operations: LLM-specific monitoring
│   ├── token_usage/            # 🔤 Operations: Token consumption tracking
│   ├── latency_tracking/       # ⏱️ Operations: Response time monitoring
│   ├── quality_monitoring/     # ✅ Operations: Output quality assessment
│   ├── safety_monitoring/      # 🛡️ Operations: Safety and alignment monitoring
│   └── cost_optimization/      # 💰 Operations: Cost tracking and optimization
├── agent_monitoring/           # 🤖 Operations: AI agent monitoring systems
│   ├── task_tracking/          # 📋 Operations: Agent task execution tracking
│   ├── goal_achievement/       # 🎯 Operations: Goal completion monitoring
│   ├── error_handling/         # 🚨 Operations: Agent error detection and recovery
│   └── learning_metrics/       # 📊 Operations: Agent learning and improvement tracking
└── user_feedback/              # 👥 Operations: User interaction and satisfaction monitoring
    ├── satisfaction/           # 😊 Operations: User satisfaction tracking
    ├── usage_patterns/         # 📈 Operations: Usage pattern analysis
    ├── feature_requests/       # 💡 Operations: Feature request management
    └── issue_tracking/         # 🐛 Operations: User issue and bug tracking

# 🧩 Supporting Infrastructure
conf/                           # ⚙️ Config: Configuration management and environment settings
├── base/                       # 🏗️ Config: Base configuration files
├── local/                      # 💻 Config: Local development overrides
├── environments/               # 🌍 Config: Environment-specific configurations
├── datasets/                   # 📋 Config: Data contracts, schemas
└── quality_gates/              # 🛡️ Config: Quality assurance configurations

tests/                          # 🧪 Testing: Comprehensive testing framework
├── unit/                       # 🔬 Testing: Unit tests for individual nodes
├── integration/                # 🔗 Testing: Integration tests for pipelines
├── data_validation/            # ✅ Testing: Data quality and schema validation tests
├── model_validation/           # 🎯 Testing: Model performance and quality tests
└── acceptance/                 # 👥 Testing: User acceptance and stakeholder tests

data_acquisition/               # 📡 Hardware: Acquisition of data and knowledge
├── sensor_systems/             # 🔌 Hardware: Sensor hardware interfaces, communication, synchronization (ROS2)
├── iot/                        # 📶 Hardware: IoT communication and messaging (MQTT, brokers)
├── kas/                        # 🧠 Hardware: Knowledge acquisition systems (ontologies, expert systems)
└── experiment_designs/         # 🔬 Hardware: Design-of-experiments for algorithm comparisons, A/B tests

pipelines/                      # 🔄 Orchestration: Automated deployment pipelines (node-pipeline frameworks)
├── hinode/                     # 🎯 Orchestration: HITnode framework pipelines (with README.md for usage)
└── kedro/                      # 🔧 Orchestration: Kedro framework pipelines (with README.md for usage)

storage/                        # 💾 Data: All storage operations (RAG, vector DBs, model storage, LLM data)
├── data/                       # 🧪 Dev/Test: Locally temporary data folder that serves sample subsets to validate nodes
├── vector_databases/           # 🔍 Prod: Vector database operations for embeddings and semantic search
│   ├── pinecone/               # 🌲 Prod: Pinecone integration nodes
│   ├── weaviate/               # 🕸️ Prod: Weaviate integration nodes
│   ├── chroma/                 # 🎨 Prod: Chroma DB integration nodes
│   ├── faiss/                  # 🔍 Prod: FAISS integration nodes
│   └── custom/                 # 🔧 Prod: Custom vector database implementations
├── model_registry/             # 📦 Prod: Model versioning, storage, and retrieval
│   ├── huggingface_hub/        # 🤗 Prod: Hugging Face model hub integration
│   ├── mlflow_registry/        # 📊 Prod: MLflow model registry
│   ├── local_registry/         # 💻 Prod: Local model storage
│   └── cloud_registry/         # ☁️ Prod: Cloud-based model storage
├── embeddings/                 # 🎯 Prod: Store and retrieve embeddings for RAG and semantic applications
│   ├── text_embeddings/        # 📝 Prod: Text embedding storage and retrieval
│   ├── image_embeddings/       # 🖼️ Prod: Image embedding storage and retrieval
│   ├── multimodal_embeddings/  # 🎭 Prod: Multi-modal embedding storage
│   └── custom_embeddings/      # 🔧 Prod: Custom embedding implementations
├── feature_store/              # 📊 Prod: Feature store management and serving
├── knowledge_graphs/           # 🕸️ Prod: Graph databases for constraints, relationships, and domain knowledge
│   ├── neo4j/                  # 🔗 Prod: Neo4j graph database integration
│   ├── rdf_stores/             # 🔗 Prod: RDF triple store integration
│   └── property_graphs/        # 🔗 Prod: Property graph implementations
├── document_stores/            # 📄 Prod: Document storage for RAG and knowledge systems
│   ├── elasticsearch/          # 🔍 Prod: Elasticsearch document storage
│   ├── mongodb/                # 🍃 Prod: MongoDB document storage
│   ├── postgres/               # 🐘 Prod: PostgreSQL with document features
│   └── custom_stores/          # 🔧 Prod: Custom document storage implementations
├── conversation_memory/        # 🧠 Prod: Conversation and session memory storage
│   ├── short_term/             # ⏱️ Prod: Short-term memory implementations
│   ├── long_term/              # 📅 Prod: Long-term memory implementations
│   ├── semantic_memory/        # 🧠 Prod: Semantic memory systems
│   └── episodic_memory/        # 📖 Prod: Episodic memory implementations
└── cache/                      # 🚀 Prod: Caching mechanisms for performance optimization
    ├── llm_cache/              # 🧠 Prod: LLM response caching
    ├── embedding_cache/        # 🎯 Prod: Embedding computation caching
    ├── retrieval_cache/        # 🔍 Prod: Retrieval result caching
    └── general_cache/          # 🔧 Prod: General purpose caching

governance/                     # 🛡️ Compliance: Compliance, audit, and policy management
├── policies/                   # 📋 Compliance: Data governance and usage policies
├── audit_trails/               # 📊 Compliance: Audit logs and compliance tracking
├── privacy/                    # 🔒 Compliance: Privacy-preserving techniques and GDPR compliance
├── risk_management/            # ⚠️ Compliance: Risk assessment and mitigation strategies
├── stakeholder_management/     # 👥 Compliance: Stakeholder communication and approval workflows
└── regulatory_compliance/      # 📋 Compliance: Regulatory compliance (EU AI Act, FDA, etc.)

utils/                          # 🔧 Utilities: Common utilities and helper functions
├── data_helpers/               # 📊 Utilities: Data manipulation and processing utilities
├── model_helpers/              # 🤖 Utilities: Model-related utility functions
├── io_helpers/                 # 📁 Utilities: Input/output operation helpers
├── security_helpers/           # 🔒 Utilities: Security and encryption utilities
├── nlp_helpers/                # 📝 Utilities: NLP utilities for text processing and analysis
├── graph_helpers/              # 🕸️ Utilities: Graph database and knowledge graph utilities
├── crisp_helpers/              # 📐 Utilities: CRISP-DM/ML(Q) workflow utilities and phase transition managers
├── llm_helpers/                # 🧠 Utilities: LLM integration and management utilities
│   ├── api_clients/            # 🔌 Utilities: LLM API client utilities (OpenAI, Anthropic, etc.)
│   ├── prompt_utils/           # 💬 Utilities: Prompt engineering and optimization utilities
│   ├── token_management/       # 🔤 Utilities: Token counting and management utilities
│   ├── response_parsing/       # 📝 Utilities: LLM response parsing and validation
│   └── cost_tracking/          # 💰 Utilities: LLM usage cost tracking utilities
├── langchain_helpers/          # 🔗 Utilities: LangChain integration and workflow utilities
│   ├── chain_builders/         # 🔗 Utilities: Chain construction utilities
│   ├── agent_utils/            # 🤖 Utilities: Agent management utilities
│   ├── tool_integrations/      # 🛠️ Utilities: Tool integration helpers
│   └── memory_utils/           # 🧠 Utilities: Memory and context management utilities
├── huggingface_helpers/        # 🤗 Utilities: Hugging Face integration utilities
│   ├── model_loaders/          # 📦 Utilities: Model loading and management utilities
│   ├── dataset_utils/          # 📊 Utilities: Dataset processing utilities
│   ├── tokenizer_utils/        # 🔤 Utilities: Tokenizer management utilities
│   └── pipeline_utils/         # 🔄 Utilities: Pipeline construction utilities
├── rag_helpers/                # 📚 Utilities: RAG system utilities and optimizations
│   ├── retrieval_utils/        # 🔍 Utilities: Retrieval optimization utilities
│   ├── chunking_utils/         # 📄 Utilities: Document chunking and processing utilities
│   ├── embedding_utils/        # 🎯 Utilities: Embedding generation and management utilities
│   └── fusion_utils/           # 🔀 Utilities: Multi-source information fusion utilities
├── mcp_helpers/                # 🔌 Utilities: Model Context Protocol utilities
│   ├── server_utils/           # 🖥️ Utilities: MCP server development utilities
│   ├── client_utils/           # 💻 Utilities: MCP client integration utilities
│   ├── protocol_utils/         # 📡 Utilities: Protocol implementation utilities
│   └── discovery_utils/        # 🔍 Utilities: Tool discovery and registration utilities
├── multimodal_helpers/         # 🎭 Utilities: Multi-modal AI utilities
│   ├── vision_utils/           # 👁️ Utilities: Computer vision utilities
│   ├── audio_utils/            # 🔊 Utilities: Audio processing utilities
│   ├── text_vision_utils/      # 📝👁️ Utilities: Text-vision integration utilities
│   └── modality_fusion/        # 🔀 Utilities: Cross-modal fusion utilities
└── agent_helpers/              # 🤖 Utilities: AI agent development utilities
    ├── planning_utils/         # 🎯 Utilities: Agent planning and reasoning utilities
    ├── execution_utils/        # 🚀 Utilities: Agent execution and control utilities
    ├── collaboration_utils/    # 🤝 Utilities: Multi-agent collaboration utilities
    └── learning_utils/         # 📊 Utilities: Agent learning and adaptation utilities

docs/                           # 📚 Documentation: Documentation and compliance records
├── api/                        # 🔌 Docs: API documentation
├── tutorials/                  # 📖 Docs: Usage examples and tutorials
├── compliance/                 # 📋 Docs: Regulatory compliance documentation
├── architecture/               # 🏗️ Docs: System architecture documentation
├── methodology/                # 📐 Docs: CRISP-ML(Q) process documentation
└── stakeholder_reports/        # 👥 Docs: Reports for business stakeholders
```

## 🔄 Node-Based Architecture

Each directory contains **nodes** - reusable components that can be combined to build ML pipelines. Nodes follow consistent interfaces for easy composition and testing.

## 🏷️ Custom [Crisp-ML](https://ml-ops.org/content/crisp-ml) table

| Activities | Subactivities and Description |
|---|---|
| **Business and Data Understanding** | - Define business objectives: requirements, constraints, success_metrics<br>- Translate business objectives into ML objectives<br>- Collect and verify data<br>- Assess the project feasibility<br>- Annotations if supervised<br>- Create POCs<br>- **GenAI:** Define generative use case (e.g., summarization, content creation) and success criteria (e.g., coherence, factuality)<br>- **Agents:** Define agent's goals, available tools (APIs, functions), and task completion metrics |
| **Data Engineering (data preparation)** | - Feature selection<br>- Data selection<br>- Class balancing<br>- Cleaning data (noise reduction, data imputation)<br>- Feature engineering (data construction)<br>- Data augmentation<br>- Data standartization<br>- **GenAI:** Curate instruction datasets for fine-tuning<br>- **GenAI (RAG):** Build and process a knowledge base for Retrieval-Augmented Generation (chunking, embedding)<br>- **Agents:** Prepare tool documentation and few-shot examples for the agent to learn from |
| **ML Model Engineering** | - Define quality measure of the model<br>- ML algorithm selection (baseline selection)<br>- Adding domain knowledge to specialize the model<br>- Model training<br>- Optional: applying trainsfer learning (using pre-trained models)<br>- Model compression<br>- Ensemble learning<br>- Model Registry: Documenting the ML model and experiments<br>- **GenAI:** Select a base Foundation Model (FM)<br>- **GenAI:** Develop system through prompt engineering, fine-tuning (e.g., LoRA), or RAG<br>- **Agents:** Design and implement the agent's reasoning loop (e.g., ReAct) and tool-use mechanisms |
| **ML Model Evaluation** | - Validate model's performance<br>- Determine robustess<br>- Increase model's explainability<br>- Make a decision whether to deploy the model<br>- Document the evaluation phase<br>- **GenAI:** Evaluate for hallucinations, toxicity, and bias (Red Teaming)<br>- **GenAI:** Use LLM-as-a-judge or human feedback (RLHF) for qualitative assessment<br>- **Agents:** Evaluate task completion success rate and tool selection accuracy |
| **Model Deployment** | - Evaluate model under production condition<br>- Assure user acceptance and usability<br>- Model governance<br>- Deploy according to the selected strategy (A/B testing, multi-armed bandits)<br>- **GenAI (RAG):** Deploy the vector database and retrieval system alongside the LLM<br>- **Agents:** Deploy the agent's reasoning engine with secure access to its tools/APIs |
| **Model Monitoring and Maintenance** | - Monitor the efficiency and efficacy of the model prediction serving<br>- Compare to the previously specified success criteria (thresholds)<br>- Retrain model if required<br>- Collect new data<br>- Perform labelling of the new data points<br>- Repeat tasks from the Model Engineering and Model Evaluation phases<br>- Continuous, integration, training, and deployment of the model<br>- **GenAI:** Monitor for prompt injection, PII leakage, and concept drift in the knowledge base<br>- **Agents:** Monitor task success rates, tool errors, and conversation logs for failures |

## 📈 Key Improvements from Industry Research & CRISP-ML(Q) + LLM/AI Alignment

- **CRISP-ML(Q) Structure**: Organized directories by CRISP-ML(Q) phases for systematic ML development
- **Business Understanding Integration**: Added dedicated `business_understanding/` and stakeholder management capabilities
- **Data Understanding Phase**: Dedicated `data_understanding/` with multi-modal analysis capabilities
- **GPAI (LLM/AI) Integration**: Comprehensive support for modern AI applications including RAG, agents, and multi-modal systems
- **GPAI Application Framework**: Complete `gpai_applications/` directory supporting prompts, agents, chains, tools, RAG, and MCP integration
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
- **Hardware Integration**: Added `data_acquisition/` for sensor systems, IoT, and knowledge acquisition systems
- **Pipeline Orchestration**: Dedicated `pipelines/` directory with support for multiple frameworks (HITnode, Kedro)

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

## 🏷️ Emoji Categorization

**Each directory is clearly marked with emojis indicating its purpose:**

```
📋 Business: Business logic and requirements
🔬 Analysis: Data analysis and exploration
📥 I/O: Input/output operations
🛠️ Transform: Data transformation and processing
🤖 Training: Model training and development
🧠 GPAI: Generative AI and LLM components
📊 Assessment: Evaluation and metrics
🚀 Production: Production deployment and serving
👁️ Operations: Monitoring and maintenance
💾 Data: Data storage and persistence
📡 Hardware: Hardware interfaces and acquisition
🔄 Orchestration: Pipeline orchestration
🔧 Utilities: Helper functions and utilities
⚙️ Config: Configuration management
🧪 Testing: Quality assurance and testing
📚 Documentation: Documentation and compliance
🛡️ Compliance: Governance and regulatory compliance
🧪 Dev/Test: Development and testing components
🌐 Prod: Production systems and operations
```

