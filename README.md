# 🧬 HInode: DSML Node Codebase Architecture Standards for Seamless Code Management and Collaboration

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
