Webscraping
===
MAIC - Spring, Week 2<br>
```
  _____________
 /0   /     \  \
/  \ M A I C/  /\
\ / *      /  / /
 \___\____/  @ /
          \_/_/
```
(Rosie is not needed!)

Prereqs:
- Install [VSCode](https://code.visualstudio.com/)
- Install [Python](https://www.python.org/downloads/)
- Ensure you can run notebooks in VSCode.

**What is webscraping?**

Are you in need of data? Maybe you want to analyze some data for insights. Or maybe you just want to train a model. In any case, you may be able to get the data you need via webscraping!

Webscraping is the process of *automatically* extracting data from websites. You can manually extract website data on your browser via "inspect," but automating this process is ideal if you need anything more than a few samples.

- Go to any website (for instance, the [MAIC](https://msoe-maic.com/) site).
- Right-click anywhere on the page. Select the "inspect" option or something labeled similarly. This is usually at the bottom of the pop-up menu.
- Note the window that opened. It contains the raw HTML (and possibly JS/CSS) site data. This is what we want to scrape automatically.
- Use the element selector at the top left of the inspect window to see the HTML of specific elements.

---

**That's cool. How can I scrape automatically?**

Download `webscraping-workshop.ipynb` to learn more, starting with extracting data from the MAIC leaderboard!