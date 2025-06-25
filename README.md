# 🧬 HITnode 
> **Modular, standards-first node library for ML & GenAI pipelines.**

<p align="center">
<img src="./docs/images/logo_1_small.png" width="200">
</p>


*Author: nmc-costa*



## 📌 Key Highlights

* **HITnode**
  Human Interface Technology-inspired, yet fully agnostic to any specific framework.

* **Standards-Driven**
  Aligns with CRISP-DM & CRISP-ML(Q) phases, LangChain patterns, Model Context Protocol (MCP) and other standards.

* **Node-Based**
  Build pipelines by snapping together reusable, well-tested nodes (`fit`/`transform`/`predict` interfaces).


---


## 🎯 VISION
- Create agnostic and modular nodes (functions || classes)
- Codebase directory architecture based on standards
- Enable autonomous generation of new projects based on these nodes instead of letting GenAI build everything from scratch
- The nodes will then be used on projects to follow a node-pipeline framework
- **NOTE**: project packages source code `src/` should follow the directory structure from this codebase. 


## 🔄 Node-Based Architecture

Each directory contains **nodes** - reusable components that can be combined to build ML pipelines. Nodes should follow consistent interfaces/packages/standards for easy composition and testing.

## 🎯 Design Principles

- **Modularity**: Each node has a single responsibility
- **Reusability**: Nodes can be used across different pipelines
- **Clarity**: Directory and file names are self-documenting
- **Scalability**: Easy to add new nodes and extend functionality

## 🛠️ Current Node-Pipeline Frameworks that work well with this standard
- **Native Python framework:**
  - [HICODE](https://github.com/nmc-costa/HIcode/blob/main/)
- **Custom 3rd party Framework:**
  -  [kedro](https://github.com/kedro-org/kedro)

## 📊 Methodology Standards
- **Data Mining**: [CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/)
- **Machine Learning**: [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml)
- **LLM Applications**: [LangChain](https://python.langchain.com/docs/concepts/)
- **Model Integration**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

## 📋 RULES

### 🎯 Node Acceptance Criteria
- **Integration**: Coordinators handle integration and merging to main branch
- **Documentation**: Nodes well documented in Jupyter notebooks with usage examples, API documentation, and integration guides
- **Testing**: Node should have tests - unit tests, dummy tests
- **Innovation**: Custom to DSML purposes or solving problems not easily handled by existing packages; 
- **Packages**: Nodes should use standard, trusted and most used packages like scikit-learn, hugging face, pandas, etc; Don't add nodes that use new and untested packages;
- **Performance**: Include benchmarks and performance considerations


### 🔧 Code Standards  
- Follow [scikit-learn](https://scikit-learn.org/stable/developers/develop.html#naming), [ML-Ops](https://ml-ops.org/content/mlops-principles#naming-conventions), [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml), [LangChain](https://python.langchain.com/docs/concepts/), and [MCP](https://modelcontextprotocol.io/) conventions
- Make them as agnostic as possible to versioning: Maintain backward compatibility or provide clear migration paths
- Minimize external dependencies, use the must trusted and used packages
- Implement configuration management through `conf/` directory

### 🚀 Git Workflow
- Create feature branch named after the node: `feature/node-name`
- Require code review before merging to main from other coordinator
- Include performance benchmarks for computationally intensive nodes
- Run automated security scanning and compliance checks
- Map contributions to CRISP-ML(Q) phases


## 🚀 Getting Started
1. **Identify CRISP-ML(Q) Phase**: Determine which phase your node belongs to (Business Understanding → Data Understanding → Data Preparation → Model Engineering → Model Evaluation → Model Deployment → Monitoring & Maintenance)
2. **Choose Application Type**: Determine if you're building traditional ML, LLM applications, RAG systems, AI agents, or multi-modal AI
3. Choose the appropriate directory for your node type based on the phase mapping and application type
4. Follow [scikit-learn](https://scikit-learn.org/stable/developers/develop.html#naming), [ML-Ops](https://ml-ops.org/content/mlops-principles#naming-conventions), [Google ML Style Guide](https://developers.google.com/machine-learning/guides/rules-of-ml), [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml), [LangChain](https://python.langchain.com/docs/concepts/), and [MCP](https://modelcontextprotocol.io/) naming conventions
5. Implement nodes with simple packages [scikit-learn style interfaces](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects) (`fit`, `transform`, `predict`) and framework-specific patterns (LangChain chains, MCP tools)
6. Set up configuration management in `conf/` directory using python, yaml or JSON and following for example [Kedro patterns](https://docs.kedro.org/en/stable/configuration/configuration_basics.html)
7. Compose nodes into pipelines as needed by copying the directory into `src/`

---

# 📁 Directory Structure

This codebase follows a node-based architecture where each directory contains reusable nodes that can be composed into ML pipelines, organized by **CRISP-ML(Q)** phases with modern AI/LLM integration. 

**NOTE**: 
- project packages source code `src/` should follow the directory structure from this codebase.
- If more folders are needed or redifinitions, do so on this structure

```
# Phase 1: Business & Data Understanding
business_understanding/          # Domain METADATA: Business objectives and stakeholder requirements processing into databases
├── requirements/               # NLP nodes to extract and embed stakeholder requirements
├── constraints/                # Parse technical/business constraints and create constraint graphs
├── success_metrics/            # Extract and formalize business KPIs and success criteria
├── domain_knowledge/           # Process domain-specific documents and create knowledge graphs
└── context_extraction/         # Extract and structure business context for LLM applications

data_understanding/              # Data exploration and automated analysis
├── exploratory_analysis/       # Automated EDA with statistical profiling
├── data_quality/               # Data quality assessment and anomaly detection for text, image, audio, and video data
├── statistics/                 # Statistical analysis and distribution modeling
├── hypothesis_generation/      # NLP-based hypothesis generation from data insights
└── metadata_extraction/        # Automated metadata extraction and cataloging

# Phase 2: Data Preparation
datasets/                        # Data I/O, ingestion, loading, and saving
├── loaders/                    # Load data from various sources (CSV, JSON, APIs, databases) and modalities (text, image, audio, video data)
├── savers/                     # Save data to different formats and destinations
├── extractors/                 # Extract data from external systems and APIs
├── validators/                 # Data quality validation and schema checking
└── huggingface_datasets/       # Hugging Face dataset integration and management

preprocessing/                   # Data cleaning, transformations, feature engineering
├── cleaners/                   # Data cleaning and quality improvement nodes
├── transformers/               # Data type conversions and transformations
├── feature_engineering/        # Feature creation, selection, and extraction
├── normalizers/                # Data normalization and scaling
├── splitters/                  # Train/validation/test data splitting
├── text_processors/            # Text preprocessing for NLP and LLM applications
├── tokenizers/                 # Tokenization nodes for various models and frameworks
└── embeddings_prep/            # Prepare data for embedding generation

# Phase 3: Model Engineering
models/                          # Training, fitting, prediction, and inference
├── trainers/                   # Model training and fitting nodes
├── predictors/                 # Prediction and inference nodes
├── tuners/                     # Hyperparameter tuning and optimization
├── architectures/              # Model architecture definitions and configurations
├── ensembles/                  # Ensemble methods and model combination
├── versioning/                 # Model versioning and comparison
├── huggingface_models/         # Hugging Face model integration and fine-tuning
├── llm_models/                 # Large language model implementations and wrappers APIs
├── embedding_models/           # Embedding model implementations (text, image, multimodal)
└── custom_architectures/       # Custom neural network architectures

# LLM Applications
llm_applications/                # LLM-powered application components
├── prompts/                    # Prompt engineering and template management
├── agents/                     # AI agent implementations and workflows
├── chains/                     # LangChain-style processing chains
├── tools/                      # AI tools and function calling
├── retrievers/                 # Information retrieval systems
├── rag_systems/                # Retrieval Augmented Generation
├── context_management/         # Context handling and memory
└── mcp_integration/            # Model Context Protocol integration

# Phase 4: Model Evaluation
evaluation/                      # Unified evaluation, metrics, and analysis
├── metrics/                    # Performance metrics (MAE, MSE, F1, AUC, BLEU, ROUGE)
├── validation/                 # Cross-validation and model validation strategies
├── testing/                    # A/B testing and statistical testing nodes
├── explainability/             # Model interpretability (SHAP, LIME, feature importance)
├── scoring/                    # Scoring predictions (thresholds, business rules)
├── comparison/                 # Model comparison and benchmarking
├── quality_assurance/          # Quality gates and acceptance criteria
└── reports/                    # Representations, plots, demos for showing to diferent users (using the data extracted during evaluation)


# Phase 5: Model Deployment 
deployment/                      # Production deployment and serving
├── serving/                    # Packaging, Ofuscation, containers, Docker images, serving endpoints
├── infrastructure/             # Infrastructure as code and deployment configs
├── user_interfaces/            # Developer, Associate, end user interfaces, StreamLit labs interfaces,
├── pipelines/                  # Automated deployment pipelines
├── rollback/                   # Rollback and recovery mechanisms
├── security/                   # Security configurations and access controls
├── agent_deployment/           # AI agent deployment systems
└── edge_deployment/            # Edge and mobile deployment

# Phase 6: Monitoring & Maintenance
monitoring/                      # Continuous monitoring and maintenance
├── performance/                # Model performance and drift monitoring
├── data_quality/              # Ongoing data quality monitoring
├── alerts/                    # Alerting and notification systems
├── maintenance/               # Model maintenance and retraining triggers
├── feedback_loops/            # Feedback collection and incorporation
├── llm_monitoring/            # LLM-specific monitoring (token usage, latency, quality)
└── agent_monitoring/          # AI agent monitoring systems

# Supporting Infrastructure
conf/                           # Configuration management and environment settings
├── base/                      # Base configuration files
├── local/                     # Local development overrides
├── environments/              # Environment-specific configurations
├── datasets/                  # data-contracts, schemas, 
└── quality_gates/             # Quality assurance configurations

tests/                          # Comprehensive testing framework
├── unit/                      # Unit tests for individual nodes
├── integration/               # Integration tests for pipelines
├── data_validation/           # Data quality and schema validation tests
└── model_validation/          # Model performance and quality tests

storage/                        # All storage operations (RAG, vector DBs, model storage)
├── vector_databases/          # Vector database operations for embeddings and semantic search
├── model_registry/            # Model store versioning, storage, and retrieval
├── embeddings/                # Store and retrieve embeddings for RAG applications
├── feature_store/             # Feature store management and serving
├── knowledge_graphs/          # Graph databases for constraints, relationships, domain knowledge
└── document_stores/           # Document storage for RAG and knowledge systems

utils/                          # Common utilities and helper functions
├── data_helpers/              # Data manipulation and processing utilities
├── model_helpers/             # Model-related utility functions
├── io_helpers/                # Input/output operation helpers
└── security_helpers/          # Security and encryption utilities

docs/                           # Documentation and compliance records
├── api/                       # API documentation
├── tutorials/                 # Usage examples and tutorials
├── compliance/                # Regulatory compliance documentation
├── architecture/              # System architecture documentation
├── methodology/               # Methodology process documentation
└── stakeholder_reports/       # Reports for business stakeholders
```


---

### 📋 GitHub Topics

```
machine-learning  crisp-ml-q  nodes  pipeline  langchain  mlops  hitnode
```
