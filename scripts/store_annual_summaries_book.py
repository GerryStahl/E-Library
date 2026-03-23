"""
store_annual_summaries_book.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stores Copilot-generated Level-2 annual summaries (summary_book) for all 20 years.
Each summary synthesizes all 3 × (up to 4) Level-1 summaries for that year.
Run from workspace root:
    python scripts/store_annual_summaries_book.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
PKL_PATH  = CACHE_DIR / "editorials_cache.pkl"
JSON_PATH = CACHE_DIR / "editorials_cache.json"
REPORT    = ROOT / "reports" / "annual_summaries_book.txt"

sys.path.insert(0, str(ROOT))
from cache.editorials_cache import EditorialsCache, EditorialSummary

TODAY = date.today().isoformat()
AUTHOR = "GitHub Copilot"
MODEL_NOTE = "GitHub Copilot (GPT-4.1), synthesizing all 3 Level-1 versions per issue"

PROMPT = (
    "You are a learning-science researcher writing for other leading researchers "
    "and advanced graduate students. You are reviewing the history of the CSCL "
    "(computer-supported collaborative learning) research field from 2006 through "
    "2025 exclusively on the basis of the 79 editorial introductions to the volumes "
    "of the 'international journal of CSCL'. "
    "Write an approximately 300-word summary of each year's volume. "
    "Focus on how the editorials described the evolving trends in the field of CSCL "
    "in the year the articles were written, especially developments concerning "
    "innovations in theory (e.g., of learning, cognition, interaction, etc.), "
    "analytic methodology (e.g., for comparing experimental cases or analyzing "
    "discourse), software technology, pedagogy, curriculum. "
    "Do not use names or technical terms that are not mentioned in this corpus. "
    "Write these summaries of the individual articles as resources for later "
    "constructing a history of the CSCL field during the 20 years covered by the journal."
)

# ── 20 Copilot-written annual summaries ────────────────────────────────────

SUMMARIES: dict[int, str] = {

2006: """\
The four inaugural issues of ijCSCL establish both the journal and the field's foundational theoretical commitments. Issue 1 announces the journal as a mark of the field's academic maturity, situating it within a trajectory from the 1989 Maratea workshop and positioning it as a global knowledge repository for an already-active research community. Issue 2 articulates a dual mission—advancing classroom collaborative learning and building the research community itself—arguing that integrating pedagogy, technology, social context, and theory is essential. Issue 3 identifies intersubjective meaning making as the field's defining theoretical construct: the primary object of analysis is how groups construct shared meaning, explicitly distinguishing CSCL from individually-focused educational research and marking a decisive shift toward group-level analysis. Issue 4 grounds the field in situated learning theory, arguing that collaborative learning in computer-mediated environments requires groups to adopt effective social practices adapted to the affordances and limitations of technology. The absent social cues of face-to-face interaction make the deliberate establishment of group social practices a distinctive design challenge. Together, the founding volume positions CSCL as a post-cognitive paradigm concerned with participation, interaction, and group meaning-making as irreducible phenomena that cannot be adequately captured by aggregating individual learning outcomes.\
""",

2007: """\
Three issues (including one double issue) mark the journal's second year. Issue 1 reflects on the journal's rapid institutional growth—nearly one hundred submissions, two hundred reviewers, forty-two Editorial Board members—and frames peer-reviewed publication as a higher standard of rigor than conference proceedings, demanding more formal argumentation and more careful theoretical positioning. The double issue timed to the CSCL 2007 conference launches the concept of flash themes—research topics that flare up within the community between conferences—and inaugurates two: scripting, defined as structured sequences of collaborative activities that guide group interaction, and argumentation, defined as discourse designed to construct and evaluate knowledge claims. Both themes probe how structured interaction shapes collaborative knowledge building. Issue 4 develops an extended metaphor of a tribal fire to theorize the dialectical relationship between individual contributions and the emergent collective knowledge-building process, framing this tension as the fundamental animating question of the CSCL field. The year's editorials collectively emphasize theoretical self-consciousness, particularly around how structured interaction and discourse patterns can be deliberately designed to support group cognition, and inaugurate a sustained commitment to revisiting contested theoretical questions across multiple future issues.\
""",

2008: """\
Four issues converge on multi-level integration as the defining pedagogical and theoretical challenge. Issue 1 introduces the SWISH principle—Split When Interaction Should Happen—proposing that effective pedagogical scripts must orchestrate transitions across individual, small-group, and classroom levels, requiring explicit design of when and how students move between collaborative and individual phases. Issue 2 uses the metaphor of the lone wolf and the pack to frame the foundational dialectic between individual cognition and group dynamics, drawing on Vygotsky to argue that characteristically human cognitive skills develop socially before being internalized individually, posing the core CSCL design challenge: how technology can mediate the intersubjective-to-individual trajectory. Issue 3 explores discourse participation through a range of theoretical lenses—communities of practice, mathematical thinking as discursive participation—while introducing computational linguistics as a potential tool for supporting the labor-intensive analysis of collaborative discourse. Issue 4 frames the upcoming CSCL 2009 conference theme of "CSCL Practices," emphasizing that the field's scope extends beyond designing technology to identifying, studying, and supporting the specific educational and professional practices through which collaborative learning actually occurs. The year marks growing attention to pedagogical design (scripting, level transitions), discourse methodology (computational analysis), and the individual-group tension that will remain central throughout the journal's history.\
""",

2009: """\
Four issues engage global societal context, practice theory, and the diversity of paradigms for conceptualizing shared knowledge. Issue 1, written in the aftermath of the 2008 US election and the global financial crisis, situates CSCL within broader societal transformation, arguing that collaborative, knowledge-building education is essential for democratic citizenship, equitable development, and a population capable of addressing complex global challenges. Issue 2 engages the "practice turn" in social theory—from explicit knowledge and social structures to embodied, materially mediated arrays of human activity—arguing that CSCL must study specific educational and professional practices rather than abstracting pre-defined variables and statistical regularities. Issue 3 reflects on the CSCL 2009 conference in Rhodes, noting the field's growing maturity: its capacity for pointed theoretical disagreement without denigrating alternative views. Issue 4 introduces four paradigms for conceptualizing shared knowledge in collaborative settings—shared individual mental representations, distributed external representations, intersubjectively constructed meaning, and socially practiced knowledge—arguing that each CSCL study implicitly adopts one paradigm with significant consequences for design and analysis. The year deepens the field's engagement with social theory and practice-based approaches while articulating the pluralism of frameworks through which shared knowledge can be conceptualized and studied.\
""",

2010: """\
Four issues mark institutional consolidation, theoretical sharpening, and a programmatic call for theoretical rigor. Issue 1 announces ijCSCL's acceptance into the ISI Web of Science Social Sciences Citation Index, validating the field's academic standing and situating this milestone within the trajectory from the 1989 Maratea workshop to the establishment of ISLS as an international institution. Issue 2 frames the issue as a prism refracting the full spectrum of CSCL work across disciplines, technologies, pedagogies, and methodologies—emphasizing that CSCL must bridge computer science and learning science while remaining open to the full range of approaches each can contribute. Issue 3 presents a theoretical diagram of group cognition, placing sequential dialogical interaction at the center and arguing that supporting group cognition requires coherent analysis at individual, group, and community levels simultaneously; such support is characterized as rare, complex, and situation-specific. Issue 4 critiques two inadequate modes of CSCL research—atheoretical naive empiricism relying on unexamined folk theories, and the mechanical application of theoretical frameworks not developed for CSCL contexts—and advocates for theoretically self-conscious inquiry that investigates and refines perspectives specifically suited to the collaborative learning phenomena the field studies. The year marks the field's most explicit call for theoretically rigorous research explicitly appropriate to CSCL.\
""",

2011: """\
Four issues engage globalization, theoretical pluralism, and new technological affordances. Issue 1 uses the 2009 PISA results—with Shanghai top-ranked in reading, science, and mathematics—to examine CSCL's relationship to Asian educational systems, challenging assumptions that Asian success reflects rote learning and asking what CSCL can offer regions undergoing educational reform emphasizing collaborative and creative problem-solving. Issue 2 celebrates theoretical and methodological pluralism, drawing on a classical Chinese poem to argue that CSCL's openness to diverse schools of thought is a scientific strength: confrontation of viewpoints and critique of established paradigms are necessary conditions for productive knowledge-building in the research community itself. Issue 3 reports on the CSCL 2011 conference in Hong Kong—the first with large-scale Asian participation, over four hundred registrants from more than thirty countries—and examines how social media tools (tweets, blogs, video streaming) extended the conference globally while also probing the limitations of mediated participation compared to physical co-presence. Issue 4 introduces tabletop computing as a flash theme, theorizing the shared physical surface—from family dinner table to tribal fire to digital tabletop—as a site of joint meaning-making, connecting new technology to deep questions about how proximity, shared orientation, and simultaneous access to a common surface enable collaborative cognition. The year marks the field's first sustained engagement with Asian CSCL contexts and with social media as both a research topic and a community-building tool.\
""",

2012: """\
Four issues sharpen the field's theoretical and institutional identity. Issue 1 presents visual models of group cognition—depicting how individual voices weave into collective meaning-making processes through sequential dialogical interaction—arguing that visual representation is essential for theorizing multi-level collaborative interaction and making explicit how individual and group planes of analysis are related. Issue 2 engages the theory of artifacts as dialectically mutually shaping cognitive processes, drawing on Hegel, Vygotsky, and activity theory to position artifacts and minds as co-constitutive rather than merely instrumental, with material form and conceptual content interpenetrating in practice. Issue 3 reports on the journal's ISI impact factor ranking—eleventh among two hundred three education journals, sixth among eighty-three information science journals—marking the field's international standing and documenting the growth of a genuinely global CSCL research community. Issue 4 addresses the theoretical challenge of connecting learning across multiple planes of analysis—individual learning, small-group cognition, and community knowledge building—arguing that all three planes are actualized in CSCL practice typically in the sequence of individual preparation, small-group collaboration, and community-level knowledge consolidation. The year marks consolidation of the multi-level theoretical framework and the growing recognition that crossing planes of analysis is both the central challenge and the distinctive contribution of CSCL research.\
""",

2013: """\
Four issues center on the theoretical and methodological challenge of learning across levels. Issue 1 introduces the CSCL 2013 conference theme—"learning across levels of space, time, and scale"—as a fundamental provocation for CSCL theory, analysis, and practice, proposing that the three planes of individual learning, group cognition, and community knowledge building may be interconnected through mechanisms not yet fully theorized or empirically described. Issue 2 introduces transactivity as a key analytical concept—reasoning that explicitly builds on the reasoning in a previous utterance—and contrasts two analytical approaches: tracing transactive sequences to individual mental states versus treating them as emergent properties of group discourse. Issue 3 reports on the CSCL 2013 conference in Madison, Wisconsin, describing pre-conference workshops on interactional resources across levels and on cyber-infrastructure for design-based research, signaling growing investment in methodological infrastructure for the field. Issue 4 returns to the journal's flash themes—argumentation, scripting, tabletop interfaces—while introducing eye-tracking technology as a new methodological tool for studying and supporting collaborative attention in shared workspaces. The year marks significant methodological expansion (transactive discourse analysis, eye-tracking) alongside the theoretical deepening of the multi-level framework that will define CSCL research through the mid-2010s.\
""",

2014: """\
Four issues engage dialogical theory, artifacts, and individual roles within collaborative groups. Issue 1 explores knowledge construction across diverse contexts—from primary school knowledge-building forums to first-year college courses to Wikipedia—examining interrelationships among individual, small-group, and community planes in line with the ICLS 2014 conference theme of practices encompassing diverse learning contexts. Issue 2 provides a comprehensive examination of dialogical theory as a foundational CSCL framework, tracing contributions of Vygotsky and Bakhtin whose social and developmental approaches challenged the individualism of Western philosophy, alongside a demonstration of the methodological pluralism characterizing the field. Issue 3 surveys five roles of artifacts in CSCL—as communication media, structures for shared representation, instructional scaffolds, targets of collaborative co-construction, and products of group knowledge building—emphasizing the centrality of material-semiotic objects in collaborative learning and knowledge creation. Issue 4 introduces innovative analytical approaches to individual roles within small-group collaboration, ranging from coding individual utterances to characterizing group-level patterns, addressing the persistent challenge of connecting micro-level interaction to macro-level outcomes. The year marks the field's strongest theoretical engagement with dialogical foundations and artifact-centered approaches, consolidating the post-cognitive theoretical framework first articulated in the founding volume.\
""",

2015: """\
Four issues mark the journal's first decade with retrospective consolidation and theoretical crystallization. Issue 1 examines knowledge artifacts—objects combining material and semiotic aspects—as formed through collaborative engagement, arguing that the interpenetration of conceptual structure and material form is central to collaborative learning, and that new knowledge artifacts are the characteristic products of effective CSCL environments. Issue 2 identifies three foundational pillars of CSCL research: the influence of social situations on collaboration, the differentiation of learning types occurring in collaborative groups, and the design of collaborative knowledge processes—presenting articles spanning undergraduate courses, professional development, and online environments. Issue 3 proposes intersubjectivity as the defining characteristic of CSCL—not a state in which individuals hold similar beliefs, but a condition achieved through productive participation in joint meaning-making discourse—arguing that CSCL groups are not arbitrary gatherings but functional collaborative units whose intersubjective capacity distinguishes them. Issue 4 surveys ten years of ijCSCL publication, tracing the field's progression from the post-cognitive paradigm revolution of the 1990s through increasingly refined theoretical, methodological, and technological contributions, characterizing the decade as one of theoretical consolidation rather than paradigm shift. The year achieves a theoretical crystallization around intersubjectivity as the field's defining concept while situating the field's decade of development within a longer history of paradigm change in learning sciences.\
""",

2016: """\
Four issues open the journal's second decade under new editorial leadership, shifting emphasis toward regulatory processes and the social-emotional dimensions of collaboration. Issue 1 marks a leadership transition, with the new editor-in-chief observing that while the field has made strong theoretical advances in understanding collaborative learning as a group-level phenomenon, the practical implementation of CSCL approaches in educational systems at scale remains an unresolved challenge. Issue 2 identifies three classical themes that continued to define CSCL research: how collaborative efforts emerge and are constituted among learners, the integration of social, emotional, and cognitive dimensions of collaboration, and the design of tools and environments supporting collaborative activity. Issue 3 announces forward-looking field initiatives, including a review study on socially shared regulation of learning examining how tools support individual and group regulatory processes, alongside design-based research on collaborative learning environments. Issue 4 presents studies spanning two research approaches: computational scaffolding through text-mining tools for heterogeneous group formation and cognitive group-awareness feedback, and detailed interactional analyses of collaborative learning support. The year opens a new phase emphasizing self-regulation, the social-emotional dimension of collaboration, and computational tools for group formation—themes that will deepen through the second decade.\
""",

2017: """\
Four issues advance "group practices" as a unifying analytical construct while engaging methodological self-reflection. Issue 1 proposes group practices—specific collaborative behaviors emerging across diverse computer-supported learning environments including online courses, museum exhibits, and elementary classrooms—as a core analytical focus for CSCL task and technology design, representing a shift from individual cognition to collective practices within student teams. Issue 2 engages with mass collaboration, adaptable scripts, complex systems theory, and collaborative writing, noting that many foundational CSCL concepts—including communities of practice—require careful specification of units of analysis before they can serve as useful analytical lenses for empirical research. Issue 3 examines technology's dual nature in collaborative learning—as both potentially divisive through enabling power asymmetries and negotiations around shared artifacts, and facilitative through structuring productive interaction—using positioning theory to analyze how students negotiate roles and relationships. Issue 4 presents three award-winning papers from the CSCL 2017 conference alongside a review paper identifying eight controversies in CSCL methodology, marking an important moment of systematic methodological self-reflection for the field. The year combines the theoretical advance of group practices as an analytical construct with methodological critique and examination of technology's ambivalent role in supporting or undermining collaborative equity.\
""",

2018: """\
Four issues examine coordination, field identity, the coupling of minds, and methodological innovation. Issue 1 problematizes collaborative processes across different contextual factors—participant differences and biases, forms of small-group regulatory processes, and how collaboration extends across multiple institutional levels—presenting research on how context shapes collaborative dynamics. Issue 2 engages directly with the question of whether CSCL constitutes a unified research community or multiple communities with divergent technologies, methodologies, and epistemologies, reporting that information processing frameworks account for only approximately eleven percent of publications while constructivism and socio-cultural theories dominate—frameworks that prove epistemologically misaligned with information processing approaches. Issue 3 articulates a comprehensive theoretical framework for CSCL as fundamentally concerned with the coupling of minds to form social systems, identifying three research strands: understanding coupling processes, designing tools to facilitate coupling, and developing analytical methods to study the outcomes. Issue 4 identifies two emerging themes in CSCL research: how students develop understanding of content beyond traditional curricula when working with digital resources, requiring new forms of scaffolding, and the need for methodologies extending beyond conventional educational frameworks to capture these complex learning trajectories. The year marks theoretical consolidation around the coupling-of-minds framework alongside growing recognition of the field's epistemological diversity as both a challenge and a resource.\
""",

2019: """\
Four issues mark the end of one editorial team's four-year tenure and reflect on technological, methodological, and institutional directions. Issue 1 examines tools and technologies for collaborative learning through two contrasting lenses—technology as agency in regulatory support versus technology as enabler of creative expression—with paired articles illustrating each approach, emphasizing that technology's role in collaborative learning is not singular but depends on design intent. Issue 2 documents a significant maturation in CSCL research: a shift from controlled laboratory settings to studying real collaborative learning settings in their full complexity, with multiple contributions examining how design elements and existing institutional structures affect core collaborative processes that had previously been studied in isolation. Issue 3, the journal's first special issue, addresses real-time orchestrational technologies—tools helping teachers monitor divergent student trajectories, coordinate group work, and provide timely guidance in expanding classroom sizes where individual attention is increasingly difficult. Issue 4 provides a retrospective editorial after four years of leadership, characterizing collaborative learning as fundamentally triadic—involving learners and computational artifacts in inseparable relation—and identifying digital infrastructures, digital tools, and the challenge of supporting collaboration at scale as defining technological directions for the field's next phase. The year marks growing complexity in both research methodology and technological scope alongside sustained attention to the teacher's orchestrational role.\
""",

2020: """\
Four issues navigate the unprecedented disruption of the COVID-19 pandemic while advancing the field's theoretical and methodological directions. Issue 1 marks an editorial leadership transition, with outgoing and incoming editors jointly articulating a vision embracing emerging technologies, big data, and computational modeling while maintaining strong theoretical foundations and renewed attention to motivational, emotional, and embodied dimensions of learning. Issue 2, published in July 2020 at the height of the pandemic's first wave, directly addresses the unprecedented forced transition to online education, arguing that CSCL has become critically important not only for educational outcomes but for societal resilience—framing collaborative interaction as a form of advocacy for human connection in times of physical distancing. Issue 3, written during the pandemic and concurrent social upheaval, emphasizes equity, diversity, and the CSCL community's values, presenting qualitative studies of collaborative sense-making processes and methodological articles examining how collaborative meaning-making unfolds across challenging social contexts. Issue 4 highlights the richness and diversity of CSCL environments through four studies examining how different configurations of collaborative support—including boundary objects linking separate classroom knowledge-building communities—enable learning across different scales and settings. The year marks the field's first sustained engagement with questions of equity and the abrupt societal demand for CSCL expertise, transforming what had been a research program into a pressing practical necessity.\
""",

2021: """\
Four issues engage multilayered design, collaboration forms across contexts, methodological maturity, and the challenge of evolving digital practices. Issue 1 examines multilayered collaboration design, arguing that effective collaborative learning requires integrating cognitive, motivational, and emotional dimensions across multiple overlapping design layers—from task structure to technology affordances to interpersonal dynamics—with each layer shaping the character of collaborative interaction. Issue 2, reflecting on the paradox that despite empirical evidence supporting collaborative learning, its benefits were not consistently demonstrated during the pandemic transition to remote learning, examines how different forms of collaboration function differently across contexts, with particular attention to how institutional and technological conditions mediate whether collaborative arrangements produce learning. Issue 3, authored by Peter Reimann, emphasizes the field's methodological maturity through four papers demonstrating sophisticated approaches to socio-technical configurations for productive talk, noting that CSCL has developed robust methods for analyzing discourse, interaction, and the material configurations supporting them. Issue 4 takes a theory-driven approach to context-aware support for collaborative discussions, examining how students' evolving digital practices—particularly through internet platforms and social media—create new challenges for collaborative learning design that existing frameworks do not yet fully address. The year marks deepening theoretical integration of motivational and emotional dimensions alongside growing sophistication in both design approaches and analytical methods.\
""",

2022: """\
Four issues address integration, relational dynamics, real-time analytics, and emotional dimensions of collaborative design. Issue 1, a special issue, proposes the systematic integration of three CSCL concepts that have typically been studied in isolation: group awareness (perceiving relevant social information about collaborating partners), collaboration scripts (cognitive schemas guiding social learning behaviors), and self-regulation—arguing that their integration is a necessary next step for both CSCL theory and the design of effective collaborative environments. Issue 2 centers on relations as foundational to collaborative learning, arguing that simply assembling learners into groups does not guarantee productive collaboration: relationships between learners, between their actions, and between learners and digital artifacts determine collaborative outcomes and shape the competencies that collaborative learning develops. Issue 3 examines real-time analytics as a mechanism for supporting perspective-taking across chasms of divergent thinking, organizing research around growth, assessment, and how real-time data can make visible collaborative dynamics that would otherwise remain opaque to teachers and learners. Issue 4 presents empirical studies on role-playing and emotions as novel CSCL design domains, integrating educational psychology's understanding of affect with CSCL's focus on collaborative processes—addressing a longstanding gap in the field's treatment of the emotional dimensions of collaboration. The year marks the most explicit integration of motivational, emotional, and relational dimensions with technological and analytical approaches that the journal had yet achieved.\
""",

2023: """\
Four issues bridge research communities, push technological frontiers, and reflect on field identity. Issue 1, written jointly with editors from the Journal of Learning Analytics, proposes nine elements essential for robust collaborative learning analytics—a framework bridging two communities that had developed largely in parallel—marking an unprecedented collaborative editorial effort aimed at establishing shared standards for analytically rigorous CSCL research. Issue 2, a special issue on extended reality technologies, examines augmented, virtual, and mixed reality environments as new contexts for collaborative and embodied learning, expanding the field's technological scope to include immersive environments where physical presence and digital interaction are combined in novel configurations. Issue 3 examines how productive collaborative processes can be enhanced through explicit scaffolding of group interaction and strategic composition of collaborative groups, presenting diverse methodological approaches to understanding when and how structured support improves collaborative outcomes. Issue 4, marking the conclusion of a four-year editorial tenure, reflects on the fundamental question of what constitutes collaboration in a period of global upheaval, AI-mediated communication, and rapidly changing digital practice, identifying the moment as a crossroads for the field's identity and theoretical foundations. The year marks the most explicit engagement with learning analytics as a partner field, with XR technologies as a new research context, and with questions of field identity under conditions of rapid technological change.\
""",

2024: """\
Four issues assert the field's maturity and open ambitious new theoretical directions. Issue 1 declares that CSCL has reached a stage of "normal science" with a well-established object of study—collaborative learning understood as a group-level phenomenon irreducible to individual cognition—while emphasizing that the field's foundational concepts continue to require theoretical refinement as new technologies and educational contexts emerge. Issue 2 identifies two major methodological threads defining current CSCL research: tool-mediated dialogue, including how teachers notice and interpret student interactions as objects for pedagogical reflection, and temporal sequence analysis, which examines how collaborative processes unfold over time rather than treating interaction as a series of discrete events. Issue 3, stimulated by the 2024 CSCL conference, articulates two primary directions for the field: understanding how collaborative pedagogies are being transformed by new technologies, and developing a unified theoretical framework that could position CSCL as simultaneously a learning science and a collaboration science. Issue 4 examines whether CSCL has fully emerged as a paradigm incommensurable with traditional educational approaches, identifying two unresolved foundational questions: whether group-level processes can be reduced to the characteristics of individual members, and whether technology fundamentally transforms collaborative processes or merely supports pre-existing ones. The year marks the most explicit engagement with questions of scientific maturity, paradigmatic identity, and unified theory.\
""",

2025: """\
Four issues engage unity and diversity, core research issues, perennial themes, and the search for a unifying theoretical construct. Issue 1 examines whether CSCL constitutes a unified scientific field or an interdisciplinary crossroads where diverse theories and methods meet, arguing that CSCL's object of study—collaborative learning understood as an irreducibly group-level phenomenon—is genuinely its own even if the field productively draws on multiple disciplinary traditions and maintains theoretical diversity. Issue 2 organizes current research around three enduring core issues: task design and its influence on knowledge construction and group regulation, dialogue and argumentation as both objects of study and vehicles of learning, and technology mediation as the distinctive context within which CSCL studies collaborative processes. Issue 3 introduces five papers addressing perennial CSCL themes—reflection through technology, collaborative learning frameworks, and the challenge of accumulating knowledge across studies—while emphasizing the field's need to balance theoretical and methodological expansion with the consolidation of established findings into a cumulative research tradition. Issue 4 proposes mechanistic explanation as a unifying foundation for CSCL research, arguing that describing how causes produce effects through underlying processes can bridge cognitive and sociocultural approaches that have coexisted uneasily throughout the field's history. The year marks the most ambitious theoretical synthesis attempt across the journal's two decades: a meta-theoretical framework proposing mechanistic explanation as a common language across CSCL's diverse theoretical communities.\
""",

}


def _truncate(obj, n=10):
    if isinstance(obj, dict):
        return {k: (" ".join(v.split()[:n]) if (k == "text" or k.endswith("_text") or k.endswith("_prompt")) and isinstance(v, str) and v else _truncate(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(i, n) for i in obj]
    return obj


def main() -> None:
    cache = EditorialsCache.load(str(PKL_PATH))
    for year in range(2006, 2026):
        cache.ensure_annual(year)

    stored = 0
    for year, text in SUMMARIES.items():
        ann = cache.get_annual(year)
        if ann is None:
            print(f"  ✗ {year}: no AnnualSummary found!")
            continue
        wc = len(text.split())
        ann.summary_book = EditorialSummary(
            summary_author          = AUTHOR,
            summary_date            = TODAY,
            summary_prompt          = PROMPT,
            summary_number_of_words = wc,
            summary_text            = text,
        )
        stored += 1
        print(f"  ✓ {year}: {wc} words stored")

    cache.save(str(PKL_PATH))
    print(f"\nPKL saved → {PKL_PATH}  ({stored} summaries stored)")

    # ── Write report ───────────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ijCSCL Annual Volume Summaries — GitHub Copilot (Book)",
        "=" * 70,
        f"Generated : {TODAY}",
        f"Model     : {MODEL_NOTE}",
        f"Input     : All 3 Level-1 versions (Claude + GPT-4o + Book) per year",
        f"Years     : {stored} / 20",
        "",
        PROMPT,
        "",
        "=" * 70,
        "",
    ]
    for year in sorted(SUMMARIES):
        ann = cache.get_annual(year)
        eds = sorted([e for e in cache.editorials if e.year == year], key=lambda e: e.issue)
        vol = eds[0].volume if eds else year - 2005
        lines.append(f"Year {year}  (Vol {vol})")
        lines.append(f"Issues  : {len(eds)}")
        lines.append(f"Titles  : " + " | ".join(e.title[:40] for e in eds))
        lines.append("")
        lines.append(ann.summary_book.summary_text)
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {REPORT}")

    # ── Re-export JSON ─────────────────────────────────────────────────────
    data = _truncate(asdict(cache))
    JSON_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029"),
        encoding="utf-8",
    )
    print(f"JSON updated   → {JSON_PATH}")


if __name__ == "__main__":
    main()
