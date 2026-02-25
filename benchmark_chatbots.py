#!/usr/bin/env python3
"""
Benchmark: chatbot_origin vs chatbot_partial_cache
=====================================================

각 모드를 **별도 프로세스**로 실행하여 공정한 비교.
(동일 프로세스에서 두 모드를 순차 실행하면 GPU/OS page cache가 공유되어 불공정)

Usage:
  # 1) Origin 모드 실행 (별도 터미널 / GPU 0)
  python benchmark_chatbots.py --mode origin --model llama --output results_origin.json

  # 2) Partial 모드 실행 (별도 터미널 / GPU 0, origin 완전 종료 후)
  python benchmark_chatbots.py --mode partial --model llama --output results_partial.json

  # 3) 결과 비교
  python benchmark_chatbots.py --compare results_origin.json results_partial.json

측정 항목:
  [Origin]  t_prefetch | t_activation | t_cache_clear
            → t_total_transition (빠름)
            → t_first_chat  ← 여기서 FULL PREFILL 발생 (느림)
            → t_total_effective = transition + first_chat

  [Partial] t_sync | t_prefetch | t_activation | t_recompute
            → t_total_transition (recompute 포함, 느림)
            → t_first_chat  ← KV 이미 계산됨 (빠름)
            → t_total_effective = transition + first_chat

핵심: t_total_effective 가 진짜 사용자 체감 비용
"""

import os
import sys
import gc
import json
import time
import argparse
import subprocess

os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 🔥 vLLM v0.8.0 버그 우회 (Prefix Caching 강제 활성화 패치)
# =========================================================
import vllm.config
# vLLM이 이 모델을 멀티모달로 착각하지 않도록 속임
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)
# =========================================================

# ============================================================================
# 모델 설정
# ============================================================================

MODELS = {
    "llama": {
        "progressive_path":   "/acpl-ssd30/7b_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    },
    "mistral": {
        "progressive_path":   "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    },
}


# ============================================================================
# 고정 대화 스크립트 (두 모드에 동일하게 적용)
# ============================================================================

# Stage 1에서 6번 대화 (KV cache 누적 — 전환 시점에 ~4200 토큰이 쌓이도록 초장문 프롬프트 사용)
# 각 프롬프트 ~500 단어(≈650 토큰), 6턴 합계 ≈ 3900 토큰 + chat template ≈ 4200 토큰
STAGE1_PROMPTS = [
    (
        "Please provide an extremely comprehensive and detailed historical account of computing, "
        "beginning with the mechanical calculators and computational devices developed in the "
        "seventeenth century by pioneers such as Blaise Pascal, who built the Pascaline in 1642, "
        "and Gottfried Wilhelm Leibniz, who designed the Stepped Reckoner and articulated the "
        "binary number system that underlies all modern digital computation. Progress through the "
        "nineteenth century with Charles Babbage's visionary but ultimately unbuilt Difference "
        "Engine and Analytical Engine, explaining why the Analytical Engine was so far ahead of "
        "its time, and why Ada Lovelace's notes on that machine earned her the title of the "
        "world's first programmer. Describe in detail how the twentieth century opened an "
        "extraordinary era of computing development, starting with the electromechanical "
        "relay-based computers of the late 1930s such as Konrad Zuse's Z3 in Germany, through "
        "the wartime computing efforts in Britain that produced Colossus at Bletchley Park under "
        "the direction of Tommy Flowers, and the American ENIAC project at the University of "
        "Pennsylvania under Eckert and Mauchly. Explain the deep theoretical foundations that "
        "Alan Turing established in his landmark 1936 paper on computable numbers and the concept "
        "of the Universal Turing Machine, and articulate how John von Neumann's 1945 draft report "
        "on the EDVAC introduced the stored-program architecture that remains the fundamental "
        "design blueprint of virtually every computer built since. Trace with technical care the "
        "invention of the transistor at Bell Laboratories in 1947 by William Shockley, John "
        "Bardeen, and Walter Brattain, and explain how this device displaced the fragile, "
        "power-hungry, unreliable vacuum tube to open an entirely new era of compact and dependable "
        "computing hardware. Continue with the invention of the integrated circuit independently "
        "by Jack Kilby at Texas Instruments in 1958 and Robert Noyce at Fairchild Semiconductor "
        "in 1959, and trace the development of the microprocessor step by step, beginning with "
        "Intel's 4004 in 1971, moving to the 8080 in 1974, and the 8086 in 1978, which "
        "established the x86 architecture that still dominates personal and server computing "
        "today. Describe the minicomputer era of the 1960s and 1970s, including the influential "
        "PDP series from Digital Equipment Corporation, which made computing accessible to "
        "universities and research laboratories that could not afford mainframes, and explain "
        "how this democratization laid the social and intellectual groundwork for what followed. "
        "Discuss the birth of the personal computer revolution with the Altair 8800 in 1975, "
        "the Apple II in 1977, the IBM PC in 1981, and the Macintosh in 1984, and explain how "
        "the graphical user interface concepts pioneered at Xerox PARC in the 1970s by researchers "
        "including Alan Kay, Butler Lampson, and Charles Thacker became the dominant paradigm "
        "for human-computer interaction that persists to this day. Address the contributions of "
        "systems software pioneers including Grace Hopper, who created the first compiler and "
        "co-designed COBOL, Dennis Ritchie and Ken Thompson, who built UNIX and the C language "
        "at Bell Labs in the early 1970s, and Linus Torvalds, who released the Linux kernel in "
        "1991 and initiated one of the most consequential open-source projects in history. "
        "Conclude with a discussion of how computing expanded far beyond its original scientific "
        "and military contexts into telecommunications, consumer electronics, automotive control "
        "systems, medical imaging, and finally into the mobile phones and ubiquitous connected "
        "devices that define contemporary life, and reflect on the geopolitical dimensions of "
        "semiconductor supply chains and the strategic importance of chip manufacturing "
        "capabilities for national and economic security in the twenty-first century."
    ),
    (
        "Provide a thorough and technically detailed account of the entire evolution of "
        "programming languages, tracing the full arc from the lowest levels of machine code "
        "and assembly language through to the diverse and sophisticated language ecosystems "
        "of today. Begin by explaining in concrete terms what machine code actually is — "
        "sequences of binary-encoded instructions that a specific processor architecture "
        "executes directly — and how assembly language introduced symbolic mnemonics to make "
        "low-level programming slightly more tractable for human programmers. Discuss the "
        "invention of the first true compiler by Grace Hopper in the early 1950s, which "
        "automatically transformed higher-level mathematical notation into machine instructions, "
        "and explain why this was considered such a radical and contested idea at the time, "
        "with many engineers insisting that no machine could write code as efficiently as "
        "a skilled human programmer. Describe in detail the development of FORTRAN by John "
        "Backus and his team at IBM in 1957, designed specifically for scientific and "
        "mathematical computing, and explain why its demonstration that compiled code could "
        "approach hand-written assembly in efficiency was so significant for the acceptance "
        "of high-level languages. Trace the subsequent history through COBOL designed for "
        "business data processing, LISP invented by John McCarthy in 1958 for symbolic "
        "computation and artificial intelligence research which introduced the concepts of "
        "recursive functions and garbage collection, ALGOL which contributed block structure "
        "and lexical scoping and introduced the Backus-Naur Form as a formal notation for "
        "language syntax, and BASIC which was explicitly designed to make programming "
        "accessible to non-specialists and students without engineering backgrounds. Explain "
        "the structured programming movement of the 1960s and 1970s championed by Edsger "
        "Dijkstra, who famously argued in a letter that the goto statement was harmful and "
        "should be eliminated from programming practice, and how Pascal was designed by "
        "Niklaus Wirth specifically to enforce structured programming discipline. Trace in "
        "detail the creation of C by Dennis Ritchie at Bell Labs in the early 1970s, which "
        "uniquely offered low-level memory control through pointer arithmetic alongside "
        "high-level abstractions, becoming the lingua franca of systems programming that "
        "it remains today. Discuss the emergence and spread of object-oriented programming "
        "through Simula in the 1960s and Smalltalk in the 1970s at Xerox PARC, and how "
        "these ideas were incorporated into C++ by Bjarne Stroustrup in the 1980s and then "
        "into Java by James Gosling at Sun Microsystems in 1995, whose write-once-run-anywhere "
        "promise via the Java Virtual Machine fundamentally reshaped enterprise software "
        "development. Describe the scripting language revolution of the 1990s with Python, "
        "Perl, Ruby, JavaScript, and PHP enabling rapid development for web applications and "
        "making programming accessible to a new generation of developers. Explain the "
        "functional programming renaissance with Haskell, OCaml, Erlang, and later Scala "
        "and Clojure influencing mainstream languages to adopt lambdas, immutable data "
        "structures, and pattern matching. Discuss the memory safety problem that motivated "
        "Mozilla Research to create Rust, which achieves memory safety without garbage "
        "collection through its ownership and borrowing type system, and conclude with "
        "reflections on the growing role of type inference, gradual typing, and AI-assisted "
        "code generation in reshaping how developers interact with programming languages."
    ),
    (
        "Describe in extensive technical depth the full evolution of computer hardware, "
        "proceeding from the earliest vacuum tube computers through transistors, integrated "
        "circuits, microprocessors, and modern billion-transistor chips to the specialized "
        "AI accelerators and emerging post-silicon technologies of today. Begin with vacuum "
        "tubes, explaining carefully how they function as electronic switches and amplifiers "
        "through the control of electron flow in an evacuated glass envelope, why they were "
        "used in early computers despite their obvious disadvantages, and what their fundamental "
        "limitations were in terms of physical size, electrical power consumption, heat "
        "generation, and catastrophic failure rates that made large-scale reliable computation "
        "extremely difficult. Explain the decisive breakthrough represented by the invention "
        "of the point-contact transistor at Bell Laboratories in December 1947, how it operates "
        "through quantum mechanical effects in doped semiconductor materials, and why the "
        "transition from vacuum tubes to solid-state transistors was so transformative, enabling "
        "computers that were smaller, cooler, cheaper, faster, and dramatically more reliable. "
        "Describe the invention of the planar integrated circuit and the development of "
        "photolithographic manufacturing processes that first enabled hundreds, then thousands, "
        "then millions, and eventually billions of transistors to be fabricated simultaneously "
        "on a single silicon wafer through repeated cycles of deposition, exposure, and etching. "
        "Explain Moore's Law as Gordon Moore originally formulated it in his 1965 paper, "
        "observing that the number of transistors on a cost-effective integrated circuit had "
        "been doubling approximately every year, and how this empirical observation became a "
        "self-fulfilling industry roadmap that drove sustained exponential improvement for six "
        "decades. Trace the evolution of CPU microarchitecture from simple single-issue "
        "in-order pipelines through the introduction of instruction pipelining, out-of-order "
        "execution, branch prediction, superscalar issue, register renaming, and aggressive "
        "speculative execution, explaining how each technique extracts more instruction-level "
        "parallelism from sequential programs. Discuss the memory hierarchy in technical detail: "
        "why different levels of cache are necessary given the fundamental and growing gap "
        "between processor speed and memory latency, how SRAM, DRAM, and NAND flash differ "
        "in their operating principles, density, and performance characteristics, and how "
        "cache coherence protocols maintain consistency across multiple processor cores. "
        "Describe the GPU revolution that NVIDIA initiated, how the massively parallel "
        "architecture of graphics processors proved ideal for general-purpose scientific and "
        "machine learning computation through CUDA, and explain why tensor operations in "
        "neural networks map so naturally to the matrix-multiply-and-accumulate hardware "
        "that GPUs provide. Discuss the physical scaling limits now confronting conventional "
        "CMOS transistors including short-channel effects, gate oxide tunneling leakage, "
        "interconnect resistance and capacitance, and thermal dissipation constraints, and "
        "what architectural responses the industry is pursuing: chiplet-based designs with "
        "advanced heterogeneous integration packaging, three-dimensional stacking of memory "
        "on logic, new transistor geometries like FinFETs and gate-all-around nanosheet "
        "transistors, and entirely new computing substrates including quantum processors "
        "that exploit superposition and entanglement to solve certain problem classes "
        "exponentially faster than any conceivable classical machine."
    ),
    (
        "What were the most significant innovations in operating systems from the earliest "
        "batch-processing monitors of the 1950s through to modern cloud-native, containerized, "
        "and serverless environments, and how has the fundamental concept of an operating "
        "system evolved in response to the radical changes in underlying hardware and "
        "application requirements over seven decades? Begin by explaining what the first "
        "operating systems actually were in concrete terms: simple batch monitors running on "
        "mainframes that accepted jobs submitted on punched cards, queued them, executed them "
        "sequentially without any notion of interactive use or resource sharing, and printed "
        "results on paper tape or line printers without any real-time feedback to the user. "
        "Describe the critical development of time-sharing systems in the early 1960s at "
        "MIT's Project MAC and the Compatible Time-Sharing System, which allowed multiple "
        "users to interact with a single large computer simultaneously through terminals "
        "distributed around a building, creating the illusion through rapid context switching "
        "that each user had a personal machine. Explain the profound importance of the "
        "creation of UNIX at Bell Laboratories by Ken Thompson and Dennis Ritchie starting "
        "in 1969, and articulate why its guiding design principles became so influential: "
        "programs should do one thing well, complex behavior should emerge from composition "
        "of small tools through pipes and filters, everything in the system should be "
        "represented as a file in a unified hierarchical namespace, and the system should be "
        "portable across hardware by writing it in a high-level language rather than assembly. "
        "Describe how UNIX spawned numerous derivatives including BSD at UC Berkeley, which "
        "contributed the crucial TCP/IP networking implementation that became foundational "
        "for the internet, and how the UNIX wars of the 1980s among competing proprietary "
        "variants eventually gave way to standardization efforts around POSIX. Trace the "
        "history of Microsoft's operating systems from CP/M-derived MS-DOS through Windows "
        "3.1's cooperative multitasking, Windows 95's introduction of preemptive multitasking "
        "and plug-and-play hardware detection for the mass market, and Windows NT which "
        "brought a security-oriented microkernel-inspired architecture with proper memory "
        "protection, user-mode drivers, and the Win32 API. Explain the development of "
        "virtual memory systems in technical detail: the distinction between physical and "
        "virtual address spaces, how multi-level page tables and hardware translation "
        "lookaside buffers enable transparent memory virtualization, how demand paging "
        "allows programs larger than available RAM to execute through page fault handling, "
        "and how address space layout randomization and executable space protection improved "
        "resistance to memory corruption attacks. Discuss modern process and thread scheduling "
        "including the Completely Fair Scheduler in Linux, real-time scheduling policies and "
        "their latency guarantees, and the unique challenges of fair scheduling across "
        "heterogeneous processor cores with different performance and efficiency profiles. "
        "Explain virtualization through hypervisors and containerization through Linux "
        "namespaces and control groups, and conclude with how the shift to cloud computing "
        "and serverless functions has dissolved the boundary between operating system and "
        "infrastructure, raising fundamental questions about what resource management and "
        "isolation mean at planetary scale."
    ),
    (
        "Analyze in complete technical depth the evolution of computer networking from the "
        "theoretical origins of packet switching through ARPANET, the TCP/IP protocol suite, "
        "the explosive growth of the commercial internet, and into the current era of "
        "software-defined networking, hyperscale data center fabrics, and global content "
        "delivery infrastructure. Begin with the concept of circuit switching as used in "
        "traditional telephony, explaining why maintaining a dedicated end-to-end physical "
        "circuit for the duration of a call is fundamentally wasteful for bursty data "
        "communication, and how Paul Baran at RAND Corporation and Donald Davies at the "
        "National Physical Laboratory in the United Kingdom independently and nearly "
        "simultaneously developed the concept of packet switching in the early 1960s as "
        "a more efficient and fault-tolerant approach. Describe the deployment of ARPANET "
        "starting in 1969 connecting four university nodes, how the initial Network Control "
        "Protocol proved insufficient as the network grew, and how the development of TCP/IP "
        "by Vint Cerf and Bob Kahn through the mid-1970s provided a layered, end-to-end "
        "architecture explicitly designed to interconnect heterogeneous underlying networks "
        "of radically different technologies. Explain the TCP/IP protocol architecture in "
        "technical detail: the network layer provides global addressing and best-effort "
        "packet forwarding through IP, the transport layer provides reliable ordered delivery "
        "with flow and congestion control through TCP or lightweight best-effort datagram "
        "delivery through UDP, and the application layer hosts the diverse protocols "
        "including HTTP, DNS, SMTP, and FTP that directly serve user applications. Discuss "
        "the development of Ethernet by Robert Metcalfe and his colleagues at Xerox PARC in "
        "1973, how it evolved through successive generations from 10 Mbps shared coaxial "
        "cable through 100 Mbps switched Fast Ethernet and Gigabit Ethernet to the 100 Gbps "
        "and 400 Gbps fabrics used in modern hyperscale data centers, and how switching "
        "replaced shared media collision domains. Trace the commercialization of the internet "
        "in the early 1990s, the invention of the World Wide Web by Tim Berners-Lee at CERN "
        "in 1989 and its explosive growth that transformed a research network into a global "
        "information infrastructure, and explain how this drove massive investment in backbone "
        "networks, Internet Exchange Points where autonomous systems peer to exchange traffic, "
        "and the submarine cable systems that carry intercontinental internet traffic. Explain "
        "the Border Gateway Protocol and how the internet's interdomain routing functions as "
        "a decentralized system of autonomous systems exchanging reachability information "
        "through policy-driven path selection, the security vulnerabilities this creates "
        "through route hijacking and prefix misconfigurations, and ongoing efforts to secure "
        "routing through Resource Public Key Infrastructure and BGPsec. Discuss how Content "
        "Delivery Networks emerged to address the fundamental latency and bandwidth challenges "
        "of serving popular content at scale by distributing it to servers physically close "
        "to end users, and conclude with the evolution of software-defined networking, network "
        "function virtualization, and the implications of 5G for industrial automation, "
        "vehicle connectivity, and the proliferation of connected devices at unprecedented scale."
    ),
    (
        "How did the personal computer revolution of the 1970s and 1980s fundamentally "
        "transform computing, and what were the cascading technical, economic, social, "
        "and cultural consequences of placing programmable computing devices in the hands "
        "of ordinary individuals for the first time in history? Begin by establishing the "
        "context of the early 1970s, when computers were exclusively large, enormously "
        "expensive mainframes or smaller but still costly minicomputers accessible only to "
        "corporations, universities, government agencies, and well-funded research "
        "institutions, and explain what changed technically to make a personal computer "
        "genuinely feasible: the integration of a complete CPU onto a single silicon chip "
        "in the form of the microprocessor, which dramatically reduced the cost and physical "
        "complexity of building a functional computer. Describe the Altair 8800, introduced "
        "on the cover of Popular Electronics in January 1975, which used Intel's 8080 "
        "processor and was sold as a kit for three hundred ninety-seven dollars, and explain "
        "how it ignited an existing hobbyist community organized around the Homebrew Computer "
        "Club in Silicon Valley and elsewhere who would form the social nucleus of the "
        "emerging personal computing industry. Explain how Bill Gates and Paul Allen wrote "
        "a BASIC interpreter for the Altair, founding Microsoft in 1975 to sell software "
        "for microcomputers, and how this established the crucial business model of software "
        "as an independent commercial product separable from the hardware on which it ran. "
        "Describe the Apple II introduced by Steve Jobs and Steve Wozniak in 1977, which "
        "was a complete and polished ready-to-use system with color graphics, a keyboard, "
        "and expansion slots, and explain how the availability of VisiCalc, the first "
        "electronic spreadsheet application, in 1979 created computing's first genuine "
        "killer application that drove substantial business adoption of personal computers "
        "purely for the practical productivity benefits. Discuss IBM's consequential decision "
        "in 1980 to enter the personal computer market rapidly using an open architecture "
        "built from commodity components, how this decision enabled a large clone industry "
        "that drove down prices and expanded the market enormously, and how Microsoft's "
        "retention of licensing rights to DOS rather than selling them outright proved to be "
        "one of the most strategically consequential business decisions in the history of "
        "technology. Explain the graphical user interface research conducted at Xerox PARC "
        "in the 1970s, including the development of the Alto personal workstation, the "
        "desktop metaphor with overlapping windows and icons, and the mouse as a pointing "
        "device, and describe how these ideas influenced the Apple Lisa in 1983 and the "
        "Macintosh in 1984 to make computing dramatically more accessible to non-technical "
        "users. Discuss the social and economic transformations enabled by personal "
        "computing: the democratization of document creation and desktop publishing, the "
        "transformation of music production and creative arts through digital audio "
        "workstations and early graphics software, the revolution in small business "
        "accounting and productivity, and the emergence of an independent software industry "
        "with thousands of vendors creating applications for niche markets that mainframe "
        "economics had made impossible. Conclude with an honest assessment of the digital "
        "divide that emerged as computer ownership correlated strongly with income, "
        "education level, and geography, creating new forms of inequality even as it "
        "democratized access to information and creative tools for those who could afford it."
    ),
]

# Stage 2에서 3번 대화 (첫 번째 = transition 직후 핵심 측정)
# max_tokens=512, max_model_len=8192: S1 6턴 후 ~3400 토큰, S2 3턴 후 ~5300 토큰
STAGE2_PROMPTS = [
    (
        "Provide a thorough explanation of machine learning fundamentals, covering supervised "
        "learning, unsupervised learning, and reinforcement learning. Include the mathematical "
        "intuition behind loss functions, gradient descent, and backpropagation. Explain how "
        "neural network depth and width affect capacity, the role of regularization and dropout, "
        "and why techniques like batch normalization and residual connections were so important "
        "for training very deep networks reliably."
    ),
    (
        "Explain the architecture and training process of transformer models in thorough detail. "
        "How does scaled dot-product attention work, and why is multi-head attention beneficial? "
        "What are the key differences between encoder-only models like BERT, decoder-only models "
        "like GPT, and encoder-decoder models like T5? Why have transformers become dominant not "
        "just in NLP but also in vision, audio, and multimodal tasks?"
    ),
    (
        "What are the major challenges in training large language models at scale? Discuss "
        "distributed training strategies including data parallelism, tensor parallelism, and "
        "pipeline parallelism. Cover memory optimization techniques such as gradient checkpointing, "
        "ZeRO optimizer states, mixed-precision training, and flash attention. What does the "
        "infrastructure stack look like for training a 100B+ parameter model?"
    ),
]

# Stage 3에서 3번 대화 (첫 번째 = transition 직후 핵심 측정)
# S2 3턴 후 ~5300 토큰, S3 3턴 후 ~7300 토큰 < 8192
STAGE3_PROMPTS = [
    (
        "What does the current trajectory of artificial intelligence development suggest about "
        "the next 10 to 20 years? Analyze trends in model scaling, multimodal and agentic "
        "capabilities, reasoning and planning, and the growing use of synthetic data and "
        "self-improvement loops. Project the likely near-term and medium-term developments, "
        "and discuss their implications for productivity, scientific discovery, and the "
        "concentration of economic and geopolitical power."
    ),
    (
        "How will AI automation affect employment across different sectors of the economy over "
        "the coming decade? Analyze which occupational categories are most exposed to automation "
        "based on task structure rather than job title. Discuss the historical precedents from "
        "previous automation waves, what new categories of work are likely to emerge, and what "
        "policy interventions — retraining programs, taxation of automation, universal basic "
        "income — are being debated and what evidence exists for their effectiveness."
    ),
    (
        "Discuss the geopolitical dimensions of the AI development race, including the US-China "
        "technology competition, export controls on advanced semiconductors, and the strategic "
        "importance of compute infrastructure and data. How are different blocs — the US, EU, "
        "China, and middle powers — approaching AI governance and regulation, and what does "
        "the fragmentation of global AI norms imply for international stability and cooperation?"
    ),
]


# ============================================================================
# 유틸리티
# ============================================================================

def drop_all_caches():
    """
    Python GC + CUDA cache + OS page cache 완전 제거.

    OS page cache 제거 이유:
    - safetensors checkpoint 파일이 이전 실행에서 RAM에 캐싱됨
    - 제거 안 하면 두 번째 실행이 디스크 I/O 없이 빠르게 로드 → 불공정
    - sudo 권한 필요: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    """
    print("\n  [Cache] Clearing all caches...")

    # 1. Python GC
    gc.collect()
    print("  [Cache]   ✅ Python GC collected")

    # 2. CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  [Cache]   ✅ CUDA allocator cache cleared")

    # 3. OS page cache
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True, timeout=15
        )
        print("  [Cache]   ✅ OS page cache dropped (sync + echo 3)")
    except subprocess.CalledProcessError as e:
        print(f"  [Cache]   ⚠️  OS page cache: sudo failed (returncode={e.returncode})")
        print("             → 수동으로 실행 필요: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    except FileNotFoundError:
        print("  [Cache]   ⚠️  OS page cache: sudo not found")
    except Exception as e:
        print(f"  [Cache]   ⚠️  OS page cache: {e}")

    time.sleep(2)  # cache settle 대기


def gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_model_handle(llm):
    """v0 엔진 progressive model handle 가져오기"""
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):
        raise RuntimeError("V1 engine detected. Set VLLM_USE_V1=0.")
    try:
        return engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError("Could not resolve v0 model handle.") from exc


def build_prompt(tokenizer, conversation: list) -> str:
    """대화 기록 → 프롬프트 문자열"""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    prompt = ""
    for msg in conversation:
        prefix = "User: " if msg["role"] == "user" else "Assistant: "
        prompt += prefix + msg["content"] + "\n"
    return prompt + "Assistant: "


def do_chat(llm, tokenizer, conversation, user_input, sampling_params):
    """
    대화 1턴 수행 + 타이밍/토큰 수 측정.
    conversation 리스트를 in-place로 업데이트.
    """
    conversation.append({"role": "user", "content": user_input})
    prompt = build_prompt(tokenizer, conversation)
    n_input = len(tokenizer.encode(prompt))

    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    t_chat = time.time() - t0

    response = outputs[0].outputs[0].text.strip()
    n_gen = len(tokenizer.encode(response))
    conversation.append({"role": "assistant", "content": response})

    return {
        "question": user_input[:60],
        "t_chat_s": round(t_chat, 3),
        "n_input_tokens": n_input,
        "n_gen_tokens": n_gen,
        "tokens_per_sec": round(n_gen / t_chat, 1) if t_chat > 0 else 0.0,
        "gpu_mem_gb": round(gpu_mem_gb(), 3),
    }


def _measure_transition_origin(llm, model, tokenizer, config, stage_key,
                                advance_fn_name, prefetch_fn_name,
                                conversation, sampling_params,
                                first_prompt):
    """
    Origin 모드 stage 전환 타이밍 측정.

    단계: prefetch → activation → cache_clear
    이후: 첫 채팅 (full prefill 발생)
    """
    tr = {}
    checkpoint_path = config[stage_key]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    prefetch_fn = getattr(model, prefetch_fn_name)
    advance_fn  = getattr(model, advance_fn_name)

    # t_prefetch: checkpoint CPU 로드 + 대기
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)

    # t_activation: GPU weight copy + alpha 변경
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # t_cache_clear: prefix cache 초기화 (다음 turn에서 full prefill 자동)
    t0 = time.time()
    llm.reset_prefix_cache()
    tr["t_cache_clear_s"] = round(time.time() - t0, 3)

    tr["t_total_transition_s"] = round(
        tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_cache_clear_s"], 3
    )
    tr["gpu_mem_after_transition_gb"] = round(gpu_mem_gb(), 3)

    # 첫 채팅 (FULL PREFILL: KV cache가 모두 비워졌으므로)
    print(f"    → t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_cache_clear={tr['t_cache_clear_s']:.3f}s | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → [First chat] FULL PREFILL (KV cache cleared)...")

    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)

    print(f"    → t_first_chat={tr['t_first_chat_s']:.3f}s  "
          f"t_total_effective={tr['t_total_effective_s']:.3f}s")

    return tr


def _measure_transition_partial(llm, model, tokenizer, config, stage_key,
                                 advance_fn_name, prefetch_fn_name,
                                 conversation, sampling_params,
                                 first_prompt):
    """
    Partial 모드 stage 전환 타이밍 측정.

    단계: sync → prefetch → activation → recompute(partial)
    이후: 첫 채팅 (KV 이미 계산됨 → 빠름)
    """
    tr = {}
    checkpoint_path = config[stage_key]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    prefetch_fn = getattr(model, prefetch_fn_name)
    advance_fn  = getattr(model, advance_fn_name)
    minimal_params = SamplingParams(temperature=0.0, max_tokens=1)

    # t_sync: GPU persistent buffer → CPU cache 동기화
    t0 = time.time()
    if hasattr(model, "model") and hasattr(model.model, "sync_persistent_cache"):
        prompt_now = build_prompt(tokenizer, conversation)
        seq_len = len(tokenizer.encode(prompt_now))
        torch.cuda.synchronize()
        model.model.sync_persistent_cache(seq_len)
        torch.cuda.synchronize()
        print(f"    → sync_persistent_cache({seq_len} tokens)")
    tr["t_sync_s"] = round(time.time() - t0, 3)

    # t_prefetch: checkpoint CPU 로드 + 대기
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)

    # t_activation: GPU weight copy + boundary 설정
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # t_recompute: partial recompute (generate max_tokens=1)
    # - boundary 이전 레이어: KV-only (캐시된 hidden states 사용, 빠름)
    # - boundary 이후 레이어: full forward (새 가중치, 정확)
    torch.cuda.synchronize()
    t0 = time.time()
    if len(conversation) > 0:
        prompt_now = build_prompt(tokenizer, conversation)
        llm.reset_prefix_cache()
        llm.generate([prompt_now], minimal_params)
    torch.cuda.synchronize()
    tr["t_recompute_s"] = round(time.time() - t0, 3)

    tr["t_total_transition_s"] = round(
        tr["t_sync_s"] + tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_recompute_s"], 3
    )
    tr["gpu_mem_after_transition_gb"] = round(gpu_mem_gb(), 3)

    print(f"    → t_sync={tr['t_sync_s']:.3f}s | "
          f"t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_recompute={tr['t_recompute_s']:.3f}s | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → [First chat] KV already updated (should be fast)...")

    # 첫 채팅 (KV 이미 partial recompute 완료 → 새 user 입력만 처리)
    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)

    print(f"    → t_first_chat={tr['t_first_chat_s']:.3f}s  "
          f"t_total_effective={tr['t_total_effective_s']:.3f}s")

    return tr


# ============================================================================
# Origin 모드 벤치마크
# ============================================================================

def run_origin(model_name: str, output_path: str):
    """
    Origin 모드 벤치마크.

    Stage 전환: reset_prefix_cache()만 호출 → 다음 chat에서 full prefill.
    사용 모듈: origin_progressive_serve/
    """
    print("\n" + "=" * 65)
    print(f"  BENCHMARK: Origin Mode  (model={model_name})")
    print("=" * 65)

    # origin_progressive_serve import
    # model_config를 먼저 import해야 progressive_model_dual_path.py의
    # hardcoded sys.path(/home/devewha/v08/...)가 우회됨
    origin_dir = os.path.join(SCRIPT_DIR, "origin_progressive_serve")
    sys.path.insert(0, origin_dir)
    import model_config as _mc_origin  # noqa: F401
    from progressive_for_causal_lm import ProgressiveForCausalLM as OriginPFCLM

    # 캐시 완전 제거 (OS page cache 포함)
    drop_all_caches()

    config = MODELS[model_name]
    model_path = config["progressive_path"]

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, OriginPFCLM)
        print(f"  Registered OriginPFCLM as: {arch}")
    except Exception as e:
        print(f"  ModelRegistry.register_model: {e}")

    # 모델 로드
    print(f"\n  Loading model from: {model_path}")
    t_load_start = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)

    print(f"  ✅ Loaded in {t_load:.1f}s  GPU={gpu_mem_gb():.2f}GB")

    results = {
        "mode": "origin",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_mem_after_load_gb": round(gpu_mem_gb(), 3),
        "stage1_chats": [],
        "stage1_to_2": {},
        "stage2_chats": [],
        "stage2_to_3": {},
        "stage3_chats": [],
    }

    conversation = []

    # ── Stage 1 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 1] {len(STAGE1_PROMPTS)} turns")
    for i, q in enumerate(STAGE1_PROMPTS):
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage1_chats"].append(r)
        print(f"    Turn {i+1}: {r['t_chat_s']:.2f}s  "
              f"({r['n_input_tokens']}→{r['n_gen_tokens']} tok, {r['tokens_per_sec']} tok/s)")

    # ── Stage 1 → 2 전환 ─────────────────────────────────────
    print(f"\n  [Stage 1 → 2] Transition...")
    tr12 = _measure_transition_origin(
        llm, model, tokenizer, config,
        stage_key="stage_b_checkpoint",
        prefetch_fn_name="prefetch_stage2",
        advance_fn_name="advance_to_stage2_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE2_PROMPTS[0],
    )
    results["stage1_to_2"] = tr12

    # ── Stage 2 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 2] {len(STAGE2_PROMPTS) - 1} more turn(s)")
    for q in STAGE2_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage2_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    # ── Stage 2 → 3 전환 ─────────────────────────────────────
    print(f"\n  [Stage 2 → 3] Transition...")
    tr23 = _measure_transition_origin(
        llm, model, tokenizer, config,
        stage_key="stage_c_checkpoint",
        prefetch_fn_name="prefetch_stage3",
        advance_fn_name="advance_to_stage3_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE3_PROMPTS[0],
    )
    results["stage2_to_3"] = tr23

    # ── Stage 3 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 3] {len(STAGE3_PROMPTS) - 1} more turn(s)")
    for q in STAGE3_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage3_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {output_path}")


# ============================================================================
# Partial 모드 벤치마크
# ============================================================================

def run_partial(model_name: str, output_path: str):
    """
    Partial 모드 벤치마크.

    Stage 전환: GPU buffer sync → prefetch → activation → partial recompute
    사용 모듈: progressive_serve/
    """
    print("\n" + "=" * 65)
    print(f"  BENCHMARK: Partial Mode  (model={model_name})")
    print("=" * 65)

    # progressive_serve import
    partial_dir = os.path.join(SCRIPT_DIR, "progressive_serve")
    sys.path.insert(0, partial_dir)
    from progressive_for_causal_lm import ProgressiveForCausalLM as PartialPFCLM

    # 캐시 완전 제거
    drop_all_caches()

    config = MODELS[model_name]
    model_path = config["progressive_path"]

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, PartialPFCLM)
        print(f"  Registered PartialPFCLM as: {arch}")
    except Exception as e:
        print(f"  ModelRegistry.register_model: {e}")

    print(f"\n  Loading model from: {model_path}")
    t_load_start = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)

    # Warmup 중 쌓인 쓰레기 데이터 제거
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print(f"  ✅ Persistent GPU buffers cleared (warmup residue removed)")

    print(f"  ✅ Loaded in {t_load:.1f}s  GPU={gpu_mem_gb():.2f}GB")

    results = {
        "mode": "partial",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_mem_after_load_gb": round(gpu_mem_gb(), 3),
        "stage1_chats": [],
        "stage1_to_2": {},
        "stage2_chats": [],
        "stage2_to_3": {},
        "stage3_chats": [],
    }

    conversation = []

    # ── Stage 1 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 1] {len(STAGE1_PROMPTS)} turns")
    for i, q in enumerate(STAGE1_PROMPTS):
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage1_chats"].append(r)
        print(f"    Turn {i+1}: {r['t_chat_s']:.2f}s  "
              f"({r['n_input_tokens']}→{r['n_gen_tokens']} tok, {r['tokens_per_sec']} tok/s)")

    # ── Stage 1 → 2 전환 ─────────────────────────────────────
    print(f"\n  [Stage 1 → 2] Transition...")
    tr12 = _measure_transition_partial(
        llm, model, tokenizer, config,
        stage_key="stage_b_checkpoint",
        prefetch_fn_name="prefetch_stage2",
        advance_fn_name="advance_to_stage2_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE2_PROMPTS[0],
    )
    results["stage1_to_2"] = tr12

    # ── Stage 2 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 2] {len(STAGE2_PROMPTS) - 1} more turn(s)")
    for q in STAGE2_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage2_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    # ── Stage 2 → 3 전환 ─────────────────────────────────────
    print(f"\n  [Stage 2 → 3] Transition...")
    tr23 = _measure_transition_partial(
        llm, model, tokenizer, config,
        stage_key="stage_c_checkpoint",
        prefetch_fn_name="prefetch_stage3",
        advance_fn_name="advance_to_stage3_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE3_PROMPTS[0],
    )
    results["stage2_to_3"] = tr23

    # ── Stage 3 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 3] {len(STAGE3_PROMPTS) - 1} more turn(s)")
    for q in STAGE3_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage3_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {output_path}")


# ============================================================================
# 결과 비교
# ============================================================================

def compare(path_a: str, path_b: str):
    """두 JSON 결과 파일을 읽어 세부 단계 시간 비교 테이블 출력"""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    label_a = f"{a['mode'].upper()} ({path_a})"
    label_b = f"{b['mode'].upper()} ({path_b})"

    def fmt_row(label, va, vb, unit="s"):
        """비교 행 포맷. None이면 N/A."""
        if va is None and vb is None:
            return f"  {label:<45}  {'N/A':>8}  {'N/A':>8}  {'':>12}"
        va_s = f"{va:.3f}{unit}" if va is not None else "N/A"
        vb_s = f"{vb:.3f}{unit}" if vb is not None else "N/A"
        if va is not None and vb is not None and va > 0 and vb > 0:
            diff = vb - va
            winner = "A" if va < vb else "B"
            pct = abs(diff) / max(va, vb) * 100
            arrow = "▼" if diff < 0 else "▲"
            diff_s = f"{arrow}{abs(diff):.3f}s ({pct:.0f}%) [{winner}↑]"
        else:
            diff_s = ""
        return f"  {label:<45}  {va_s:>8}  {vb_s:>8}  {diff_s}"

    W = 80
    print("\n" + "=" * W)
    print(f"  COMPARISON")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"  Model: {a['model']}")
    print("=" * W)
    print(f"  {'Metric':<45}  {'A':>8}  {'B':>8}  {'Delta (winner↑)':>20}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*20}")

    # ── 로드 시간 ──
    print(fmt_row("Model load time", a["t_load_s"], b["t_load_s"]))

    # ── Stage 1 채팅 ──
    print(f"\n  {'── Stage 1 Chats ──':}")
    chats_a = a.get("stage1_chats", [])
    chats_b = b.get("stage1_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        label = f"  Turn {i+1} chat"
        va = ca["t_chat_s"] if ca else None
        vb = cb["t_chat_s"] if cb else None
        print(fmt_row(label, va, vb))

    def print_transition(label_12, ta, tb):
        print(f"\n  {'── ' + label_12 + ' ──':}")

        # sync (partial only)
        va = ta.get("t_sync_s", 0.0)
        vb = tb.get("t_sync_s", 0.0)
        print(fmt_row("  t_sync [GPU→CPU buffer, partial only]", va, vb))

        print(fmt_row("  t_prefetch [ckpt CPU load]",
                      ta.get("t_prefetch_s"), tb.get("t_prefetch_s")))
        print(fmt_row("  t_activation [GPU weight copy]",
                      ta.get("t_activation_s"), tb.get("t_activation_s")))

        # cache_clear (origin only)
        va_cc = ta.get("t_cache_clear_s", 0.0)
        vb_cc = tb.get("t_cache_clear_s", 0.0)
        print(fmt_row("  t_cache_clear [origin only]", va_cc, vb_cc))

        # recompute (partial only)
        va_rc = ta.get("t_recompute_s", 0.0)
        vb_rc = tb.get("t_recompute_s", 0.0)
        print(fmt_row("  t_recompute [partial only]", va_rc, vb_rc))

        print(fmt_row("  t_total_transition",
                      ta.get("t_total_transition_s"), tb.get("t_total_transition_s")))

        n_in_a = ta.get("first_chat_n_input", "?")
        n_in_b = tb.get("first_chat_n_input", "?")
        print(fmt_row(f"  t_first_chat [KEY] "
                      f"(A:{n_in_a}tok, B:{n_in_b}tok)",
                      ta.get("t_first_chat_s"), tb.get("t_first_chat_s")))

        print(fmt_row("  ★ t_total_effective [transition+first_chat]",
                      ta.get("t_total_effective_s"), tb.get("t_total_effective_s")))

    print_transition("Stage 1 → 2 Transition",
                     a.get("stage1_to_2", {}), b.get("stage1_to_2", {}))

    # Stage 2 chats
    print(f"\n  {'── Stage 2 Chats ──':}")
    chats_a = a.get("stage2_chats", [])
    chats_b = b.get("stage2_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        print(fmt_row(f"  Turn {i+1} chat",
                      ca["t_chat_s"] if ca else None,
                      cb["t_chat_s"] if cb else None))

    print_transition("Stage 2 → 3 Transition",
                     a.get("stage2_to_3", {}), b.get("stage2_to_3", {}))

    # Stage 3 chats
    print(f"\n  {'── Stage 3 Chats ──':}")
    chats_a = a.get("stage3_chats", [])
    chats_b = b.get("stage3_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        print(fmt_row(f"  Turn {i+1} chat",
                      ca["t_chat_s"] if ca else None,
                      cb["t_chat_s"] if cb else None))

    # ── 핵심 요약 ──
    print("\n" + "=" * W)
    print("  ★ KEY INSIGHT (t_total_effective = transition + first_chat)")
    print(f"  {'Transition':<20}  {'A':>10}  {'B':>10}  Winner")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

    for label, key in [("Stage 1→2", "stage1_to_2"), ("Stage 2→3", "stage2_to_3")]:
        ta = a.get(key, {})
        tb = b.get(key, {})
        va = ta.get("t_total_effective_s")
        vb = tb.get("t_total_effective_s")
        if va is not None and vb is not None:
            winner = "A" if va < vb else "B"
            saving = abs(va - vb)
            pct = saving / max(va, vb) * 100
            print(f"  {label:<20}  {va:>9.2f}s  {vb:>9.2f}s  "
                  f"[{winner}] faster by {saving:.2f}s ({pct:.0f}%)")

    print("\n  NOTE:")
    print("  - t_first_chat in Origin = FULL PREFILL (all tokens recomputed)")
    print("  - t_first_chat in Partial = only new user tokens processed")
    print("  - t_recompute in Partial = boundary layers only (KV snapshot for front layers)")
    print("=" * W)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chatbot_origin vs chatbot_partial_cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run origin mode
  python benchmark_chatbots.py --mode origin --model llama --output results_origin.json

  # Run partial mode (separate process, after origin finishes)
  python benchmark_chatbots.py --mode partial --model llama --output results_partial.json

  # Compare results
  python benchmark_chatbots.py --compare results_origin.json results_partial.json
        """,
    )
    parser.add_argument(
        "--mode", choices=["origin", "partial"],
        help="Which chatbot mode to benchmark"
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="llama",
        help="Model to use (default: llama)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output JSON path for benchmark results"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
        help="Compare two result JSON files"
    )

    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])

    elif args.mode:
        if not args.output:
            parser.error("--output is required with --mode")

        print(f"\n{'='*65}")
        print(f"  Chatbot Benchmark")
        print(f"  Mode:  {args.mode}")
        print(f"  Model: {args.model}")
        print(f"  GPU:   {torch.cuda.get_device_name(0)}")
        print(f"  Out:   {args.output}")
        print(f"{'='*65}")

        if args.mode == "origin":
            run_origin(args.model, args.output)
        else:
            run_partial(args.model, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
