# LLM Reasoning & Personality Knowledge Graph

## Project Overview
This project extracts structured knowledge from unstructured text and models personality traits of individuals in a Knowledge Graph (KG). It identifies people, organizations, their relationships, and personal characteristics such as traits and activities. The project demonstrates how NLP techniques and LLM-assisted reasoning can generate actionable insights from textual data.

**Key Features:**
- Named Entity Recognition (NER) to detect people and organizations using spaCy.  
- Trait extraction (adjectives) and activity extraction (verbs) for each person.  
- Relationship modeling, e.g., `works_at` between people and organizations.  
- JSON output representing the knowledge graph.  
- Interactive visualization using PyVis.  
- Evaluation of extraction using precision, recall, and F1-score.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/KG_Project.git](https://github.com/yourusername/KG_Project.git)
    ```
2.  **Navigate to the project folder:**
    ```bash
    cd KG_Project/code
    ```
3.  **Install Dependencies:**
    ```bash
    pip install spacy pyvis
    python -m spacy download en_core_web_lg
    # Fallback models (en_core_web_trf or en_core_web_sm) are used if en_core_web_lg is unavailable.
    ```
4.  **Run the main script:**
    ```bash
    python import spacy.py
    ```

The script successfully generates two key output files in the `outputs/` folder:
- `knowledge_graph.json` → structured data of the KG.
- `knowledge_graph.html` → interactive graph visualization in your browser.

---

## Evaluation

The extraction is evaluated using **precision**, **recall**, and **F1-score** by comparing predicted entities, traits, activities, and relationships with a manually defined ground truth.

### Example Results (Per Person)

| Person | Type | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| Aisha | Traits | 0.33 | 0.50 | 0.40 |
| Aisha | Activities | 1.00 | 1.00 | 1.00 |
| Aisha | Works\_at | 1.00 | 1.00 | 1.00 |
| Lina | Traits | 1.00 | 1.00 | 1.00 |
| Lina | Activities | 1.00 | 1.00 | 1.00 |
| Lina | Works\_at | 1.00 | 1.00 | 1.00 |
| Omar | Traits | 1.00 | 1.00 | 1.00 |
| Omar | Activities | 1.00 | 1.00 | 1.00 |
| Omar | Works\_at | 0.00 | 0.00 | 0.00 |
| Zain | Traits | 0.50 | 1.00 | 0.67 |
| Zain | Activities | 1.00 | 1.00 | 1.00 |
| Zain | Works\_at | 1.00 | 1.00 | 1.00 |

### Average Performance

- **Traits:** P=0.71, R=0.88, F1=0.77
- **Activities:** P=1.00, R=1.00, F1=1.00
- **Works\_at:** P=0.75, R=0.75, F1=0.75

---

## Visualization

The KG is visualized interactively using **PyVis**.

- **People** → `ellipse` nodes
- **Organizations** → `box` nodes
- **Traits** → `red dots`
- **Activities** → `green dots`
- **Edges** are explicitly labeled (`has_trait`, `does`, or `works_at`)

Open the generated `knowledge_graph.html` file in a web browser to explore the interactive graph.

---

## Limitations & Future Work

| Area | Limitation | Future Improvement |
| :--- | :--- | :--- |
| **Extraction** | Some traits or activities may be missed or misclassified. | Implement **transformer-based NER** (like BERT) for higher accuracy. |
| **Relationships** | Only `works_at` relationships are explicitly modeled. | Include `collaborates_with`, `reports_to`, or **project-based relationships**. |
| **Visualization** | Basic node design. | Richer visualizations with **logos/images** or dynamic updates. |

---

## Acknowledgments

- **SpaCy** for powerful NLP capabilities.
- **PyVis** for interactive graph visualization.
- **LLMs** for valuable workflow and design guidance.
