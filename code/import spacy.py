import spacy
import json
from collections import defaultdict

# Load spaCy's NLP model for entity extraction (try large model first, fallback to others)
try:
    nlp = spacy.load("en_core_web_lg")  # Large model
    print("Using en_core_web_lg for NER.")
except OSError:
    try:
        nlp = spacy.load("en_core_web_trf")  # Advanced transformer-based model
        print("en_core_web_lg not available; using en_core_web_trf.")
    except OSError:
        nlp = spacy.load("en_core_web_sm")  # Basic model as fallback
        print("en_core_web_trf not available; using en_core_web_sm.")

# Example text
text = """
Aisha is a creative and open-minded software engineer at TechNova. 
Recently, Aisha and Lina organized a hackathon to encourage teamwork and inspire innovation among interns. 
Lina, a calm and analytical product manager at CloudSync, ensures every project stays on schedule. 
Her colleague Omar, a disciplined data scientist, works closely with her on data-driven solutions. 
Meanwhile, Zain, an energetic marketing strategist at VisionHub, supported the event by promoting it across social media.
"""

# Initialize containers
people = set()
organizations = set()
person_data = defaultdict(lambda: {"traits": [], "activities": [], "sentences": []})
relationships = []

# Process sentences# Process sentences
for sent in nlp(text).sents:
    sent_text = sent.text.strip()
    if not sent_text:
        continue
    doc = nlp(sent_text)
    
    # Detect entities
    sent_people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    sent_orgs = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
    
    for p in sent_people:
        people.add(p)
        person_data[p]["sentences"].append(sent_text)
        
        # Extract traits from noun chunks
        for chunk in doc.noun_chunks:
            if p in [ent.text for ent in chunk.ents if ent.label_ == "PERSON"]:
                traits = [tok.text for tok in chunk if tok.pos_ == "ADJ"]
                person_data[p]["traits"].extend(traits)
        
        # Extract adjectives in the sentence near the person
        for tok in doc:
            if tok.pos_ == "ADJ" and any(ent.text == p for ent in doc.ents if ent.label_ == "PERSON"):
                person_data[p]["traits"].append(tok.text)
        
        # Extract activities (verbs in the sentence)
        activities = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]
        person_data[p]["activities"].extend(activities)
        
        # Check for works_at relationships
        for org in sent_orgs:
            relationships.append([p, "works_at", org])
            organizations.add(org)


# Remove duplicates in traits and activities
for p in person_data:
    person_data[p]["traits"] = list(set(person_data[p]["traits"]))
    person_data[p]["activities"] = list(set(person_data[p]["activities"]))

# Build final JSON
kg = {
    "people": list(people),
    "organizations": list(organizations),
    "person_data": person_data,
    "relationships": relationships
}

# Save to file
with open("knowledge_graph.json", "w", encoding="utf-8") as f:
    json.dump(kg, f, indent=2)

print(json.dumps(kg, indent=2))

# --- Evaluation ---

# Ground truth
ground_truth = {
    "people": ["Aisha", "Lina", "Omar", "Zain"],
    "organizations": ["TechNova", "CloudSync", "VisionHub"],
    "traits": {
        "Aisha": ["creative", "open-minded"],
        "Lina": ["calm", "analytical"],
        "Omar": ["disciplined"],
        "Zain": ["energetic"]
    },
    "activities": {
        "Aisha": ["organize", "encourage", "inspire"],
        "Lina": ["ensure", "organize", "encourage", "inspire", "stay"],
        "Omar": ["work", "drive"],
        "Zain": ["support", "promote"]
    },
    "relationships": [
        ("Aisha", "works_at", "TechNova"),
        ("Lina", "works_at", "CloudSync"),
        ("Zain", "works_at", "VisionHub")
    ]
}

# Evaluation function
def evaluate_extraction(pred, true):
    pred_set = set(pred)
    true_set = set(true)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Predicted relationships
predicted_relationships = kg["relationships"]

# Initialize sums for averages
sum_metrics = {"Traits": [0,0,0], "Activities": [0,0,0], "Works_at": [0,0,0]}
num_people = len(kg["person_data"])

# Print table header
print(f"{'Person':<10} | {'Type':<12} | {'Precision':<8} | {'Recall':<8} | {'F1':<8}")
print("-"*55)

# Evaluate for each person
for person in kg["person_data"]:
    # Traits
    p, r, f = evaluate_extraction(kg["person_data"][person]["traits"], ground_truth["traits"][person])
    sum_metrics["Traits"][0] += p
    sum_metrics["Traits"][1] += r
    sum_metrics["Traits"][2] += f
    print(f"{person:<10} | {'Traits':<12} | {p:<8.2f} | {r:<8.2f} | {f:<8.2f}")

    # Activities
    p, r, f = evaluate_extraction(kg["person_data"][person]["activities"], ground_truth["activities"][person])
    sum_metrics["Activities"][0] += p
    sum_metrics["Activities"][1] += r
    sum_metrics["Activities"][2] += f
    print(f"{person:<10} | {'Activities':<12} | {p:<8.2f} | {r:<8.2f} | {f:<8.2f}")

    # Works_at relationships
    true_rels = [rel for rel in ground_truth["relationships"] if rel[0] == person]
    pred_rels = [tuple(rel) for rel in predicted_relationships if rel[0] == person]
    p, r, f = evaluate_extraction(pred_rels, true_rels)
    sum_metrics["Works_at"][0] += p
    sum_metrics["Works_at"][1] += r
    sum_metrics["Works_at"][2] += f
    print(f"{person:<10} | {'Works_at':<12} | {p:<8.2f} | {r:<8.2f} | {f:<8.2f}")

# Print average metrics
print("-"*55)
for metric_type in ["Traits", "Activities", "Works_at"]:
    avg_p = sum_metrics[metric_type][0] / num_people
    avg_r = sum_metrics[metric_type][1] / num_people
    avg_f = sum_metrics[metric_type][2] / num_people
    print(f"{'Average':<10} | {metric_type:<12} | {avg_p:<8.2f} | {avg_r:<8.2f} | {avg_f:<8.2f}")

# --- Visualization ---
from pyvis.network import Network
import json

# Load your knowledge graph JSON
with open("knowledge_graph.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

# Create PyVis network
net = Network(height="800px", width="100%", notebook=False, bgcolor="#FFFFFF", font_color="black")

# Add organization nodes
for org in kg["organizations"]:
    net.add_node(org, label=org, color="#ffcc00", shape="box")

# Add people nodes (without duplication)
for person, pdata in kg["person_data"].items():
    net.add_node(person, label=person, color="#00ccff", shape="ellipse")
    
    # Add edges for traits
    for trait in pdata["traits"]:
        trait_node = f"{person}_{trait}"  # make a unique node for trait
        net.add_node(trait_node, label=trait, color="#ff6666", shape="dot")
        net.add_edge(person, trait_node, label="has_trait", color="#ff6666")
    
    # Add edges for activities
    for act in pdata["activities"]:
        act_node = f"{person}_{act}"  # make a unique node for activity
        net.add_node(act_node, label=act, color="#66ff66", shape="dot")
        net.add_edge(person, act_node, label="does", color="#66ff66")

# Add edges for works_at relationships (existing)
for rel in kg["relationships"]:
    source, relation, target = rel
    net.add_edge(source, target, label=relation, color="#C1C1C1")

    net.add_edge(source, target, label=relation, color="#AAA9A9")

# Generate and save interactive HTML
net.write_html("knowledge_graph.html")
print("✅ Knowledge graph saved as knowledge_graph.html")



