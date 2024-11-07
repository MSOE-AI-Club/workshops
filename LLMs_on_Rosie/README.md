LLMs on Rosie
===
MAIC - Fall, Week 10<br>
```
  _____________
 /0   /     \  \
/  \ M A I C/  /\
\ / *      /  / /
 \___\____/  @ /
          \_/_/
```
(Run on Rosie)

---

**Methods of running LLMs**

Theory is great and all, but how can one actually run an LLM?

[Llama.cpp](https://github.com/ggerganov/llama.cpp) is a solution for running LLMs locally!  
It could only run Llama initially, but it can now run most open source LLMs.
Fun fact: llama.cpp does not depend on any machine learning or tensor libraries (like Tensorflow or Pytorch, each of which are hundereds of megabytes); it was written from scratch in C/C++.

Another solution for running LLMs locally: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).

---

But how does one use Llama.cpp?

To use it on Rosie, do the following:

- Connect to MSOE via VPN.
- Start a VSCode server on the [Rosie dashboard](https://dh-ood.hpc.msoe.edu/pun/sys/dashboard).
- Make sure you have the python extension installed.
  - If you're unfamiliar, take a look at this [extension installation guide](https://code.visualstudio.com/docs/editor/extension-marketplace)
  - Search for and install the extension named `Python`. It should be the one with around 3 or 4 million downloads.
- Open the command palette (F1 or Ctrl+Shift+P).
- Search for and select `Python: Select Interpreter`
- Choose the first option: `Enter interpreter path...`
- Choose the first option again: `Find...`
- Copy & paste `/data/ai_club/team_3_2024-25/team3-env-py312-glibc/bin/python`. Press enter.
- Ignore all errors that may pop up (they shouldn't matter)
- Download the `llm-rosie-workshop.ipynb` file and drag it into VSCode. Put it somewhere in your file browser on the left, and open it up.
- In the upper-right corner, in the notebook toolbar, select a kernel.
- Choose `select another kernel`. (This may be skipped if you haven't used VSCode through the Rosie dashboard.)
- Choose `Python Environments`
- Choose `team3-env-py312-glibc`
- Now you can run code. In the future you should only have to start from selecting a kernel.
- Import `lamma_cpp` to see if things are working:

---

Download the notebook to continue.