# 🪶 The Grimoire Static Site

This directory contains a lightweight, zero-build static site that beautifully renders the main `README.md` as an interactive portfolio.

## 🛠️ How It Works

Instead of maintaining a separate JSON file or database, **the `README.md` acts as the single source of truth**.

1. When the page loads, `script.js` uses the browser's `fetch()` API to download the raw text of the `README.md`.
2. It parses the markdown, slicing it by the `---` horizontal rules.
3. Using basic string matching, it extracts the project titles, dates, descriptions, themes, skills, and links.
4. It converts any inline markdown (`**bold**`, `*italics*`, `[links](url)`) into proper HTML.
5. Finally, it dynamically renders the "Spell Cards" into the DOM and populates the tag filters.

Because there is no build step, **any time you update the `README.md` and push to GitHub, the site updates automatically.**

## 🚀 How to Run Locally

Because the JavaScript uses the `fetch()` API to read the `README.md` file, you **cannot** simply double-click the `index.html` file to open it. Modern web browsers block `fetch()` requests on local files (`file://` protocol) for security reasons.

To test the site locally, you need to spin up a quick local web server. Python makes this incredibly easy:

1. Open your terminal or command prompt.
2. Navigate to this `grimoire` folder:
   ```bash
   cd path/to/grimoire
   ```
3. Start the built-in Python HTTP server:
   ```bash
   python -m http.server 8000
   ```
4. Open your web browser and go to:
   **[http://localhost:8000](http://localhost:8000)**

## 🌐 Deployment (GitHub Pages)

Deploying the site is effortless:
1. Ensure `index.html`, `styles.css`, and `script.js` are committed to your repository.
2. Go to your repository settings on GitHub.
3. Navigate to **Pages**.
4. Select the branch you want to deploy from (usually `main` or `master`) and save.
5. GitHub will automatically serve your `index.html` file, and the JavaScript will successfully fetch your `README.md`!
