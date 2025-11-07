# 🪶 Grimoire
*A record of the projects, experiments, and creations crafted along my path as a computer science student.*

This repository gathers my academic and personal projects — from foundational coursework to independent explorations.
Each entry represents a small spell of learning, discovery, or problem-solving.

---

### 🕯️ HN Monitor — High-Scoring Story Sentinel

**Jul 2023 – Dec 2025 · Personal Project**

> *A quiet watcher in the background: guarding attention by revealing only what truly rises above the noise of Hacker News.*

An **Android** application that passively monitors **Hacker News** and surfaces only high-impact stories, allowing users to stay informed without repeatedly visiting the site or getting pulled into endless scrolling.

The app periodically fetches *Top* and *Best* stories via the **official Hacker News API**, persists qualifying entries locally, and presents them in a clean, distraction-free interface.

Background execution enables reliable periodic fetches even when the app is closed. **Optional push notifications** alert the user when new stories exceed the chosen threshold.

**Skills:** Android Development, Kotlin

**Repository:** 🔗 [hn-monitor](https://github.com/este6an13/HN-Monitor-App)

---

### 🧪 Experimental Design for LLM Code Generation Evaluation

**Mar 2025 · Independent Research**

> *Where language models become collaborators — refining thought into computation.*

An experimental framework for evaluating collaborative code generation with large language models using verifiable rewards. This project explores how LLM agents collaborate to iteratively generate and optimize code within a controlled experimental design. By varying the number of agents and the optimization strategy (iterations or time limit), we evaluate collaboration dynamics through objective signals — specifically, whether generated programs execute correctly and how efficiently they run. Execution time and correctness serve as verifiable criteria, enabling rigorous assessment of LLM performance and reliability without relying on subjective judgments. Although the statistical power achieved in this pilot was limited, the project demonstrates the potential of combining experimental design with verifiable evaluation to better understand collaborative dynamics in LLMs.

**Skills:** Large Language Models (LLMs), AI Agents, Experimental Design, Statistical Data Analysis

**Repository:** 🔗 [llm-code-refinery](https://github.com/este6an13/llm-code-refinery)


---

### 🤖 RAG-Based University Chatbot

**Mar 2024 – Jul 2024 · Volunteering**

> *A voice born from knowledge — guiding students through halls of data and learning.*

A chatbot that uses Retrieval Augmented Generation (RAG) to answer questions for the Faculty of Engineering community at the National University of Colombia. An Open-Source Software initiative for the National University of Colombia.

**Skills:** Chatbots, Retrieval-Augmented Generation (RAG), Generative AI, Open-Source Software

**Repository:** 🔗 [reprebot](https://github.com/Represoft/reprebot)

---

### 💳 LLM-Powered Optical Character Recognition (OCR) for Bank Check Information Extraction

**Jan 2024 – Feb 2024 · Commission**

> *Where language models read between the lines — turning ink and handwriting into structured understanding.*

Software to extract key information from printed and handwritten text on bank checks, using object detection techniques, cloud ML services and Retrieval Augmented Generation (RAG). The solution provides enhanced transparency by reporting confidence levels in the OCR results.

**Skills:** Optical Character Recognition (OCR), Generative AI, Machine Learning, RAG

**Repository:** 🔗 [checks-ocr](https://github.com/este6an13/checks-ocr)

---

### 🫁 Medical Report Generation with Pre-Trained Medical Transformers

**Oct 2023 – Dec 2023 · Universidad Nacional de Colombia**

> *When vision meets language — machines learn to describe what they see within the human form.*

Implemented an **encoder–decoder transformer architecture** to automatically **generate diagnostic reports** from **chest X-ray images**, leveraging **pre-trained models** specialized in the **medical domain**.

**Repository:** Available Soon

**Course:** Computer Vision

**Skills:** Machine Learning, NLP, PyTorch, TensorFlow, Generative AI, Transformers

---

### 🧫 Neural Networks to Detect Sjögren Syndrome in Salivary Gland Images

**Oct 2023 – Nov 2023 · Universidad Nacional de Colombia**

> *Teaching machines to see what only specialists once could — patterns hidden in the silence of cells.*

Implementation and comparison of different neural networks models: shallow architectures (with feature extraction based on machine vision techniques) and transfer learning leveraging visual transformers and deep CNN architectures.

**Repository:** Available Soon

**Course:** Neural Networks

**Skills:** Neural Networks, Deep Learning, PyTorch, TensorFlow, Machine Learning

---

### 🌀 Adaptive Reservoir Computing with Kuramoto Oscillators

**Jun 2023 · Universidad Nacional de Colombia**

> *When oscillators think — exploring computation at the edge of synchrony.*

Implemented and evaluated an **adaptive reservoir computing algorithm** built on a network of **Kuramoto phase oscillators**, exploring its ability to predict **nonlinear time-series**. Generated benchmark sequences (NARMA10, MG17, MSO12), integrated their dynamics (RK4/Euler), and trained the system using ridge regression. Analyzed prediction error, synchronization, spectral radius effects, and the evolution of the adaptive coupling matrix.

Although the results deviated from the reference study, the project exposed critical implementation and numerical sensitivities—highlighting the need to re-examine the original implementation to achieve a faithful and stable reproduction of the model.

**Repository:** 🔗 [rc-kuramoto-sad-rc](https://github.com/este6an13/rc-kuramoto-sad-rc)

**Course:** Computational Physics

**Skills:** Python, Reservoir Computing, Kuramoto Model, Ridge Regression, Time-Series Modeling, Dynamical Systems

---

### 🐜 Profiling, Optimization, and Parallelization of Logistics Route Planning

**Mar 2023 – Jun 2023 · Universidad Nacional de Colombia**

> *When colonies of algorithms march in parallel — finding order in the chaos of routes.*

Optimized and parallelized an **Ant Colony Optimization (ACO)** algorithm to find near-optimal solutions to the **Travelling Salesman Problem (TSP)**.
Implemented in **C++ (MPI)** and **Python**, focusing on **profiling**  and **scalability** in **high-performance computing** environments running on **Linux clusters**.

**Repository:** Available Soon

**Course:** High Performance Computing (HPC)

**Skills:** C++, High-Performance Computing (HPC), Python, Linux, Scientific Computing

---

### 🎯 Retrieval-Augmented Generation (RAG) as a Software Development Lifecycle Tool

**Jun 2023 · Universidad Nacional de Colombia**

> *Where language models become architects — weaving code from knowledge and intent.*

Implementation of RAG to support the software development lifecycle (SDLC) by automatically generating user stories, test scenarios, requirement specifications and troubleshooting solutions from product backlogs and technical documentation of a software engineering product.

**Repository:** Available Soon

**Course:** Machine Learning

**Skills:** Retrieval-Augmented Generation (RAG), Generative AI, Software Engineering

---

### 🔐 Differentially Private Recommender System

**May 2023 · Universidad Nacional de Colombia**

> *Guarding secrets in the data shadows — teaching machines to recommend without revealing.*

Implemented a **privacy-preserving recommender system** applying **Differential Privacy** to an **SVD-based matrix factorization** model.
Introduced **Laplace noise mechanisms** to user-item ratings and model gradients, ensuring data protection while maintaining predictive utility.
Trained and evaluated the model on the **MovieLens 100k** dataset using **K-fold cross-validation**, analyzing the trade-off between privacy budget (ϵ) and accuracy (RMSE).
Compared results against a baseline SVD model to study convergence and generalization under controlled privacy budgets.

**Repository:** Available Soon

**Course:** Cybersecurity

**Skills:** Python, Differential Privacy, SVD, Matrix Factorization, Recommender Systems, Machine Learning

---

### 🌲 Random Forest Model to Predict Employee Attrition

**Nov 2022 – Dec 2022 · Universidad Nacional de Colombia**

> *Uncovering the patterns behind people’s choices — where data meets human behavior.*

Random Forest ML model to detect and identify the factors that lead to employee attrition rates in the Sales, Human Resources, and R&D departments of a company.

**Repository:** Available Soon

**Course:** Intro to AI

**Skills:** Machine Learning, Scikit-learn, Python

---

### ♨️ Heat Transfer Simulation in a Furnace

**Sep 2022 · Universidad Nacional de Colombia**

> *Mapping invisible fire — modeling heat diffusion through walls of a furnace with numerical precision.*

Developed a **numerical solver** to compute the **steady-state temperature distribution** in a furnace wall using the **finite difference method** and **triangular factorization (LU decomposition)**.
Implemented the full algorithm in **GNU Octave**, discretizing the 2D heat equation and applying **convective and symmetry boundary conditions** to construct and solve large systems of linear equations.
Generated heat maps visualizing the temperature field within the furnace based on user-defined geometry and material parameters, in an **interdisciplinary collaboration with Chemical Engineering students**.

**Repository:** Available Soon

**Course:** Numerical Methods

**Skills:** Numerical Analysis, Finite Difference Method, LU Decomposition, GNU Octave, MATLAB, Simulation

---

### 💞 Cloud-Native Microservices Dating App

**Feb 2022 – Jul 2022 · Universidad Nacional de Colombia**

> *Where systems connect as seamlessly as people do.*

Developed a **web and mobile dating platform** built on a **cloud-native microservices architecture**, designed to explore interoperability across diverse programming languages and frameworks.
Implemented services in **Vue.js**, **ASP.NET Core**, **Go**, and **Python**, integrating **SQL** and **NoSQL** databases under **REST** and **GraphQL** APIs.
Deployed and orchestrated using **Docker** and **Kubernetes** on **Google Cloud Platform (GCP)**, achieving scalable service communication and high availability.

**Repository:** Available Soon

**Course:** Software Architecture

**Skills:** Docker, Kubernetes, GraphQL, ASP.NET Core, Golang, Python, Vue.js, Google Cloud Platform (GCP), Microservices Architecture

---

### 🏥 Medical Records System

**Aug 2021 – Jan 2022 · Universidad Nacional de Colombia**

> *Building a bridge between data and care — one record at a time.*

Developed a **web application system** for managing **medical records** in healthcare providers (**IPSs**), designed under an **MVC architecture** to ensure modularity and maintainability.

**Repository:** Available Soon

**Course:** Software Engineering II

**Skills:** Django, Vue.js, PostgreSQL, Git, Full-Stack Development

---

### 🧮 Virtual John von Neumann Machine over a Heterogeneous Ad-Hoc Network

**Nov 2021 · Universidad Nacional de Colombia**

> *Reimagining a classic architecture — distributing a single mind across many machines.*

Developed a **virtual computing system** that composes a single **John von Neumann (JvN) machine** from five **heterogeneous nodes** connected through a simulated **ad-hoc network**.
Each node implements a distinct **microarchitecture** (stack-in-memory vs. stack-separate) managed by local hypervisors, enabling the virtual machine to transparently combine and access distributed components — **memory, ALU, control unit, stack, and I/O**.
The system supports a **22-instruction microarchitecture**, a full **assembler/linker/loader toolchain**, and can execute programs such as **prime detection** and **GCD computation** across nodes.
Implemented entirely in **Python**, exploring concepts in **virtualization**, **distributed systems**, and **architecture emulation**.

**Repository:** Available Soon

**Course:** Compilers

**Skills:** Python, Virtualization, Distributed Systems, Computer Architecture

---

### 🛵 Home Delivery App

**Feb 2021 – Jul 2021 · Universidad Nacional de Colombia**

> *Connecting small businesses and neighbors — bringing local commerce to the digital doorstep.*

Developed a **full-stack web application** to support home delivery services for small neighborhood stores and businesses.
The backend was implemented in **Java** using **Spring**, with **MongoDB** as the **NoSQL** database for scalable data storage.
The client was built in **Flutter (Dart)** to provide a responsive, cross-platform interface.

**Repository:** Available Soon

**Course:** Software Engineering I

**Skills:** Java, Spring, MongoDB, Dart, Flutter, Full-Stack Development

---

### 🛡️ XSS Attack Detection System with Machine Learning

**Jun 2021 – Jul 2021 · Universidad Nacional de Colombia**

> *Where security meets intelligence — teaching machines to spot deception in the web’s hidden layers.*

Developed a **cloud-based system** for detecting **Cross-Site Scripting (XSS)** attacks using **machine learning** and **web traffic analysis**.
Implemented a **proxy-based MITM architecture** to capture and decrypt HTTPS traffic, analyzing HTML and JavaScript content for malicious patterns.
Built a **Flask** server for real-time classification using a **Random Forest** model, integrated with an **ASP.NET Core MVC** dashboard and **SignalR** for live alerts — all deployed on **Microsoft Azure**.

**Repository:** Available Soon

**Course:** Computer Networks

**Skills:** Python, Flask, ASP.NET Core MVC, SignalR, Azure, Machine Learning, Cybersecurity

---

### 📱 Sinograms Mobile App

**Aug 2020 – Jan 2021 · Universidad Nacional de Colombia**

> *Bridging language and algorithms — a tool to explore the logic behind Chinese characters.*

Developed an **Android** application for consulting Chinese characters (*sinograms*), implementing **AVL Trees**, **Max Heaps**, and the **Rabin–Karp** algorithm for efficient data retrieval and pattern matching.
Focused on applying advanced data structures to real-world use cases in language processing.

**Repository:** Available Soon

**Course:** Data Structures

**Skills:** Java, Android Development, Algorithms, Data Structures

---

### 🗃️ Fixed Assets Inventory Management System

**Jul 2020 – Aug 2020 · Universidad Nacional de Colombia**

> *Building order from chaos — a system to track and tame institutional assets.*

Developed a fixed assets inventory management application using the **Oracle APEX** low-code platform.

**Repository:** Available Soon

**Course:** Database Systems

**Skills:** SQL, Oracle APEX, Data Modeling

---

### 🎮 AdventureMath RPG Video Game
**Aug 2019 – Jan 2020 · Universidad Nacional de Colombia**

> *A quest to merge learning and play — crafting a world where math becomes an adventure.*

One level of a role-playing game designed to teach fundamental math concepts to children.

**Repository:** Available Soon

**Course:** Object-Oriented Programming

**Skills:** C#, Unity

