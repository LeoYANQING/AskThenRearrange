# §3 HHI Coding Methodology — Literature Grounding Report

**Purpose**: Identify established methodological frameworks to support PrefQuest's §3 dialogue coding methodology for IJHCS submission.

**Date generated**: 2026-04-21

---

## Executive Summary

The paper's reliance on Braun & Clarke (thematic analysis) alone as methodological justification is **critically insufficient** — and internally inconsistent, because Braun & Clarke's reflexive thematic analysis **explicitly rejects inter-rater reliability**, yet the paper reports Cohen's κ. The fix does not require changing the coding scheme; it requires reframing the methodological justification in established frameworks.

**Three anchors to add:**
1. **Mayring (2000, 2015)** — qualitative content analysis → replaces Braun & Clarke
2. **ISO 24617-2 / DAMSL** — dialogue act annotation standards → justifies hierarchical multidimensional coding
3. **Sequence analysis / CA** — pattern identification from coded sequences

---

## Area 1: Dialogue Act Annotation Frameworks

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Allen & Core (1997). "Coding Dialogs with the DAMSL Annotation Scheme" | AAAI FSS | Gold standard for multidimensional dialogue annotation |
| Jurafsky et al. (1997). Switchboard-DAMSL | Computational Linguistics | Applied DAMSL to large corpus; 42 core dialogue acts |
| Bunt et al. (2012, 2020). ISO 24617-2 | ISO / LREC | Current international standard for dialogue annotation |
| Traum (2010). "20 Questions on Dialogue Act Taxonomies" | UMIACS | Design choices in dialogue act schemes |

### Core Finding

ISO 24617-2 and DAMSL establish that:
- Multidimensional, hierarchical coding is **standard practice**
- Utterances simultaneously carry multiple communicative functions
- Formal coding rules and IRR measurement are expected

### Citation Framing for §3

> "Following established dialogue act annotation standards (ISO 24617-2: Bunt et al., 2012; DAMSL: Allen & Core, 1997), we developed a hierarchical, multidimensional coding scheme in which each utterance is coded at three levels: knowledge type, interactional intent, and sequential pattern."

---

## Area 2: Qualitative Content Analysis (PRIMARY REPLACEMENT FOR BRAUN & CLARKE)

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Mayring (2000, 2015). "Qualitative Content Analysis" | Forum QS | Systematic rule-based coding with IRR measurement |
| Neuendorf (2002, 2017). "The Content Analysis Guidebook" | SAGE | Standard reference; κ ≥ 0.80 widely acceptable |
| Jordan & Henderson (1995). "Interaction Analysis" | J. Learning Sciences | Sequential multi-turn analysis of dialogue |
| Braun & Clarke (2006, 2019) | SAGE | ⚠️ **Explicitly rejects IRR — unsuitable as sole justification when κ is reported** |

### Critical Issue

Braun & Clarke's reflexive thematic analysis is **philosophically incompatible** with reporting inter-rater reliability. Their 2019 paper explicitly states that IRR is inappropriate for reflexive thematic analysis. Citing them while also reporting κ signals methodological confusion to reviewers.

**Mayring is the correct replacement**: designed for systematic, reproducible dialogue coding; mandates IRR; supports both deductive (existing categories) and inductive (emergent categories) procedures.

### Citation Framing for §3

> "We applied Mayring's (2000) qualitative content analysis procedure, which provides systematic, rule-based analysis with explicit inter-rater reliability requirements. Unlike reflexive thematic analysis (Braun & Clarke, 2006), qualitative content analysis is specifically designed for transparent, reproducible coding where inter-rater agreement is a core quality criterion (Neuendorf, 2017). We combined deductive category application (Level 1 knowledge types, grounded in established cognitive psychology frameworks) with inductive category formation (Level 2 interactional intents, emergent from the data)."

---

## Area 3: HHI → Robot/Agent Design Pipeline

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Riek (2012). "Wizard of Oz Studies in HRI: A Systematic Review" | J. Human-Robot Interaction | Establishes HHI coding → agent design as standard HRI methodology |
| Marge et al. (2017). "Applying WoZ to Multimodal HRI Dialogue" | AAAI FSS | Dialogue collection → analysis → agent design pipeline |
| Bohus & Horvitz (2009). "Models for Multiparty Engagement" | SIGDIAL | Bridge from HHI observation to computational models |

### Core Finding

The pipeline **observe human dialogue → systematically code → extract patterns → implement as agent behavior** is established practice in HRI. Riek (2012) reviewed 54 HRI WoZ studies and formalized reporting guidelines for this methodology. This legitimizes §3 as Step 1 of a principled design process, not an informal "inspiration."

### Citation Framing for §3

> "Following established practice in human-robot interaction research (Riek, 2012; Marge et al., 2017), we conducted an HHI study to extract dialogue patterns as the basis for agent design. The systematic coding of human asker-role behavior (§3) constitutes the first step of an HHI-to-agent pipeline: observe and code natural human questioning strategies → identify recurring patterns → formalize as dialogue policies → implement in the PrefQuest agent (§4)."

---

## Area 4: Question Taxonomy and Classification

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Graesser & Person (1994). "Question Asking During Tutoring" | Am. Educational Research J. | Comprehensive taxonomy of question types in teaching dialogue |
| Lehnert (1978). "The Process of Question Answering" | Erlbaum | 13-category question taxonomy from cognitive science |
| Cakmak & Thomaz (2012). "Designing Robot Learners that Ask Good Questions" | HRI | 3-type taxonomy for robot learning questions (label, demonstration, feature) |
| Li & Roth (2002). "Learning Question Classifiers" | COLING | Hierarchical question taxonomy for NLP |

### Core Finding

**AO/PE/PS are interactional intent categories, not semantic content categories** — these are orthogonal dimensions, and both are established in the literature.

AO ≈ Cakmak & Thomaz's (2012) "demonstration queries" (task-coordination)
PE ≈ Cakmak & Thomaz's "feature queries" / Graesser's "verification/elaboration"
PS ≈ Graesser's "assertion questions" / dialogue repair questions

The key clarification for reviewers: the paper's categorization is about **communicative goal** (why the questioner is asking), not **semantic content** (what the question is about). This needs to be stated explicitly.

### Citation Framing for §3

> "Our interactional intent dimension (Action-Oriented, Preference-Eliciting, Preference-Summarizing) is complementary to semantic question taxonomies (Graesser & Person, 1994; Lehnert, 1978). While semantic taxonomies classify questions by content type, interactional intent focuses on the communicative goal the questioner pursues within the teaching context. The AO category corresponds to task-coordination questions analogous to Cakmak & Thomaz's (2012) demonstration queries; PE corresponds to exploratory preference-seeking questions found in preference elicitation dialogue (Christakopoulou et al., 2016)."

---

## Area 5: Sequential Pattern Analysis in Dialogue

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Solomon et al. (2022). "Conversational Motifs in Supportive Interactions" | J. Social & Personal Relationships | Sequence analysis methodology for deriving multi-turn dialogue patterns |
| Sacks, Schegloff, & Jefferson (1974). Turn-taking | Language | Conversation Analysis foundations — adjacency pairs, sequential implicativeness |
| Various HMM/transition matrix papers | ACL/SIGDIAL | Quantitative methods for dialogue sequence analysis |

### Core Finding

There is established methodology for the move from **individual coded utterances → higher-level dialogue patterns**:
- Sequence analysis (identifying frequent intent sequences / "motifs")
- Transition matrix / lag-sequential analysis (significant consecutive transitions)
- Qualitative inspection of intent distribution across dialogue position (beginning / middle / closing)

The current §3 performs the third method (examining intent temporal distribution) but doesn't name or justify the methodology. Citing Solomon et al. (2022) and/or Conversation Analysis literature legitimizes this move.

### Citation Framing for §3

> "To identify dialogue patterns (Level 3), we applied sequence analysis methodology (Solomon et al., 2022). For each dyad, we extracted the ordered sequence of interactional intents and examined: (1) which intent types appeared in the opening, middle, and closing segments; (2) whether preference-related intents clustered at the beginning or distributed throughout; and (3) dominant consecutive intent transitions. Two researchers independently classified each of the 28 dialogue sequences into candidate patterns, achieving κ = [TODO]; disagreements were resolved through discussion."

---

## Area 6: Teaching / Tutoring Dialogue Annotation

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Chi et al. (2001). "Learning from Human Tutoring" | Cognitive Science | Multi-level dialogue annotation for teaching contexts |
| Graesser et al. (2004). AutoTutor corpus | Behavior Research Methods | Multi-level annotated tutoring corpus (dialogue acts + affect + strategy) |
| Litman & Forbes-Riley (2006). ITSPOKE | ACL/SIGDIAL | Multi-dimensional tutoring dialogue coding |

### Core Finding

The tutoring dialogue field provides a direct analogy: household organization teaching is structurally similar to instructional tutoring (one person knows, one person learns through questions). Tutoring dialogue researchers use **multi-level coding** (local dialogue acts + global strategy) — exactly what PrefQuest does.

### Citation Framing for §3

> "Our study of dialogue in household organization teaching draws methodological parallels with tutoring dialogue research (Chi et al., 2001; Graesser & Person, 1994). As in tutoring, the dialogue involves asymmetric knowledge (the teacher knows their preferences, the learner does not) and is driven by the learner's questions. We adopt the multi-level annotation approach used in tutoring corpora (e.g., AutoTutor: Graesser et al., 2004), coding dialogue at both the local (individual question intent) and global (sequence-level pattern) levels."

---

## Area 7: Inter-Rater Reliability Standards

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Artstein & Poesio (2008). "Inter-Coder Agreement for Computational Linguistics" | Computational Linguistics | Comprehensive survey of κ, α, π — recommends Krippendorff's α |
| Landis & Koch (1977). "Observer Agreement for Categorical Data" | Biometrics | κ interpretation guidelines (substantial ≥ .61, almost perfect ≥ .81) |
| Neuendorf (2017). "The Content Analysis Guidebook" | SAGE | Content analysis standard: κ ≥ .80 widely acceptable, κ ≥ .90 universally |
| Krippendorff (2004). "Content Analysis" | SAGE | Krippendorff's α — more appropriate than κ for many dialogue coding tasks |

### Core Finding

**Community standards for dialogue annotation:**
- κ ≥ .70 = minimum acceptable
- κ ≥ .80 = strongly preferred (content analysis standard)
- κ ≥ .90 = near-universal acceptance

**Recommendation**: Report both Cohen's κ (per-level, per-category) and consider Krippendorff's α for hierarchical codes. Artstein & Poesio (2008) is the key citation for justifying α over κ in multidimensional coding.

### Citation Framing for §3

> "Inter-rater reliability was assessed using Cohen's κ (Cohen, 1960), following Landis & Koch's (1977) interpretation scale, with κ ≥ .70 as the minimum acceptable threshold for dialogue annotation (Artstein & Poesio, 2008). We report agreement for each coding level separately, as reliability may vary across the knowledge-type, intent, and pattern levels. [Report actual κ values with 95% CI]."

---

## Area 8: Preference Elicitation Dialogue Analysis

### Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| Christakopoulou et al. (2016). "Towards Conversational Recommender Systems" | KDD | PE questions in CRS — directly validates PE category |
| Zhang & Balog (2020). "Evaluation of CRS" | Survey | Preference elicitation as distinct dialogue phase |
| Sepliarskaia et al. (2018). "Preference Elicitation as Classification" | RecSys | Annotating user preference expressions in dialogue |

### Core Finding

Preference elicitation through dialogue questions is a recognized subfield. The PE category directly maps to the "preference elicitation phase" in CRS research. Citing this literature grounds PrefQuest's PE construct in peer-reviewed dialogue research.

---

## Recommended Revised §3 Structure

```
§3.1 Overview and Study Design
Frame as Step 1 of HHI-to-agent pipeline (Riek 2012).

§3.2 Participants and Procedure
[Largely unchanged]

§3.3 Coding Methodology
Lead paragraph: "We applied qualitative content analysis (Mayring, 2000, 2015) ..."
Sub-paragraph: "Following dialogue act annotation standards (ISO 24617-2; DAMSL) ..."
Sub-paragraph: "Inter-rater reliability was assessed via Cohen's κ (Artstein & Poesio, 2008) ..."

§3.3.1 Level 1: Knowledge-type Annotation
[Unchanged — K-type + Wilson already cited]

§3.3.2 Level 2: Intent-level Coding  
Add: connection to Graesser, Cakmak & Thomaz, Christakopoulou et al.
Clarify: "interactional intent ≠ semantic content" distinction

§3.3.3 Level 3: Pattern Identification
Add: sequence analysis methodology (Solomon et al. 2022)
Add: describe the process (opening/middle/closing segment analysis + transition analysis)

§3.4 Pattern Distribution
[Unchanged]

§3.5 From Patterns to Strategies
[Largely unchanged]
```

---

## New Bibliography Entries Needed

```bibtex
@article{mayring2000qualitative,
  author    = {Mayring, Philipp},
  title     = {Qualitative Content Analysis},
  journal   = {Forum Qualitative Sozialforschung / Forum: Qualitative Social Research},
  volume    = {1},
  number    = {2},
  year      = {2000},
  url       = {https://doi.org/10.17169/fqs-1.2.1089}
}

@book{neuendorf2017content,
  author    = {Neuendorf, Kimberly A.},
  title     = {The Content Analysis Guidebook},
  edition   = {2nd},
  publisher = {SAGE Publications},
  year      = {2017}
}

@article{artstein2008inter,
  author    = {Artstein, Ron and Poesio, Massimo},
  title     = {Inter-Coder Agreement for Computational Linguistics},
  journal   = {Computational Linguistics},
  volume    = {34},
  number    = {4},
  pages     = {555--596},
  year      = {2008}
}

@inproceedings{allen1997damsl,
  author    = {Allen, James F. and Core, Mark G.},
  title     = {Coding Dialogs with the {DAMSL} Annotation Scheme},
  booktitle = {AAAI Fall Symposium on Communicative Action in Humans and Machines},
  year      = {1997}
}

@article{bunt2012iso,
  author    = {Bunt, Harry and Alexandersson, Jan and Choe, Jae-Woong and Fang, Alex Chengyu and Hasida, K{\^o}iti and Petukhova, Volha and Popescu-Belis, Andrei and Traum, David},
  title     = {{ISO} 24617-2: {A} Semantically-Based Standard for Dialogue Annotation},
  booktitle = {Proceedings of LREC},
  year      = {2012}
}

@article{riek2012wizard,
  author    = {Riek, Laurel D.},
  title     = {Wizard of {Oz} Studies in {HRI}: A Systematic Review and New Reporting Guidelines},
  journal   = {Journal of Human-Robot Interaction},
  volume    = {1},
  number    = {1},
  pages     = {119--136},
  year      = {2012}
}

@article{jordan1995interaction,
  author    = {Jordan, Brigitte and Henderson, Austin},
  title     = {Interaction Analysis: Foundations and Practice},
  journal   = {Journal of the Learning Sciences},
  volume    = {4},
  number    = {1},
  pages     = {39--103},
  year      = {1995}
}

@article{graesser1994question,
  author    = {Graesser, Arthur C. and Person, Natalie K.},
  title     = {Question Asking During Tutoring},
  journal   = {American Educational Research Journal},
  volume    = {31},
  number    = {1},
  pages     = {104--137},
  year      = {1994}
}

@inproceedings{christakopoulou2016towards,
  author    = {Christakopoulou, Konstantina and Radlinski, Filip and Hofmann, Katja},
  title     = {Towards Conversational Recommender Systems},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {815--824},
  year      = {2016}
}

@article{artstein2008irrCompLing,
  author    = {Artstein, Ron and Poesio, Massimo},
  title     = {Inter-Coder Agreement for Computational Linguistics},
  journal   = {Computational Linguistics},
  volume    = {34},
  number    = {4},
  pages     = {555--596},
  year      = {2008}
}

@article{solomon2022motifs,
  author    = {Solomon, Denise Haunani and Weber, Kathleen M. and others},
  title     = {Using Sequence Analysis to Identify Conversational Motifs in Supportive Interactions},
  journal   = {Journal of Social and Personal Relationships},
  volume    = {39},
  number    = {7},
  year      = {2022}
}

@article{landis1977kappa,
  author    = {Landis, J. Richard and Koch, Gary G.},
  title     = {The Measurement of Observer Agreement for Categorical Data},
  journal   = {Biometrics},
  volume    = {33},
  number    = {1},
  pages     = {159--174},
  year      = {1977}
}
```

---

## Impact Assessment

| Area | Current §3 | After Reframing |
|---|---|---|
| Primary framework | Braun & Clarke ❌ (rejects IRR) | Mayring ✅ (mandates IRR) |
| Dialogue coding standard | Not cited ❌ | ISO 24617-2 / DAMSL ✅ |
| Pattern identification | No method cited ❌ | Sequence analysis ✅ |
| Intent category grounding | Not grounded ❌ | Graesser, Cakmak & Thomaz, CRS ✅ |
| IRR contextualization | κ reported without framing ❌ | Artstein & Poesio, Landis & Koch ✅ |
| Study framing | Standalone observation ❌ | HHI-to-agent pipeline (Riek) ✅ |

**Estimated reviewer skepticism reduction:** HIGH → LOW for NLP/dialogue reviewers; MODERATE → LOW for HCI reviewers.
