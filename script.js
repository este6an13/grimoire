document.addEventListener('DOMContentLoaded', () => {
    // Theme logic
    const themeToggle = document.getElementById('theme-toggle');
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');

    // Check saved theme
    const savedTheme = localStorage.getItem('grimoire-theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcons(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        document.body.setAttribute('data-theme', newTheme);
        localStorage.setItem('grimoire-theme', newTheme);
        updateThemeIcons(newTheme);
    });

    function updateThemeIcons(theme) {
        if (theme === 'dark') {
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        } else {
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        }
    }

    // State
    let projects = [];
    let allTags = new Set();
    let activeTags = new Set();
    let searchQuery = '';

    // DOM Elements
    const grid = document.getElementById('spell-grid');
    const selectedTagsContainer = document.getElementById('selected-tags');
    const searchInput = document.getElementById('search-input');
    const tagSearchInput = document.getElementById('tag-search-input');
    const tagSuggestions = document.getElementById('tag-suggestions');

    // Fetch and parse README
    async function loadSpells() {
        try {
            // Add cache busting query param to ensure fresh fetch
            const response = await fetch('README.md?t=' + new Date().getTime());
            if (!response.ok) throw new Error('Could not fetch README.md');
            const markdown = await response.text();
            
            projects = parseReadme(markdown);
            extractTags();
            renderSelectedTags();
            renderProjects();
        } catch (error) {
            console.error('Error loading spells:', error);
            if (window.location.protocol === 'file:') {
                grid.innerHTML = `<div class="loading">It seems you opened this file directly. You must run a local web server (e.g., <code>python -m http.server</code>) to allow fetching the README.md.</div>`;
            } else {
                grid.innerHTML = `<div class="loading">Failed to decipher the Grimoire. Ensure README.md exists and is accessible.</div>`;
            }
        }
    }

    // Basic markdown parser tailored to the README structure
    function parseReadme(md) {
        // Normalize line endings for cross-platform compatibility
        md = md.replace(/\r\n/g, '\n');
        
        // Split by horizontal rules
        const sections = md.split(/\n---\n/);
        const parsedProjects = [];

        // Skip the first section (header)
        for (let i = 1; i < sections.length; i++) {
            const section = sections[i].trim();
            if (!section) continue;

            const project = {
                title: '',
                meta: '',
                quote: '',
                description: [],
                themes: [],
                skills: [],
                craft: [],
                repoLink: '',
                repoText: '',
                urlLink: '',
                urlText: ''
            };

            const lines = section.split('\n');
            let inDescription = false;

            for (let j = 0; j < lines.length; j++) {
                const line = lines[j].trim();
                if (!line) continue;

                if (line.startsWith('### ')) {
                    project.title = line.replace('### ', '');
                    inDescription = false;
                } else if (line.startsWith('**') && line.endsWith('**') && !line.includes('Themes:') && !line.includes('Skills:') && !line.includes('Craft:') && !line.includes('Repository:') && !line.includes('URL:') && !line.includes('Course:')) {
                    project.meta = line.replace(/\*\*/g, '');
                    inDescription = false;
                } else if (line.startsWith('> *') && line.endsWith('*')) {
                    project.quote = line.replace('> *', '').replace('*', '');
                    inDescription = true; // Description usually follows quote
                } else if (line.startsWith('**Themes:**')) {
                    const themesStr = line.replace('**Themes:**', '').trim();
                    project.themes = themesStr.split(',').map(t => t.trim().replace(/\.$/, ''));
                    inDescription = false;
                } else if (line.startsWith('**Skills:**')) {
                    const skillsStr = line.replace('**Skills:**', '').trim();
                    project.skills = skillsStr.split(',').map(s => s.trim().replace(/\.$/, ''));
                    inDescription = false;
                } else if (line.startsWith('**Craft:**')) {
                    const craftStr = line.replace('**Craft:**', '').trim();
                    project.craft = parseCraft(craftStr);
                    inDescription = false;
                } else if (line.startsWith('**Repository:**')) {
                    const repoStr = line.replace('**Repository:**', '').trim();
                    const linkMatch = repoStr.match(/\[(.*?)\]\((.*?)\)/);
                    if (linkMatch) {
                        project.repoText = linkMatch[1];
                        project.repoLink = linkMatch[2];
                    } else {
                        project.repoText = repoStr.replace(/🔗/g, '').trim();
                    }
                    inDescription = false;
                } else if (line.startsWith('**URL:**')) {
                    const urlStr = line.replace('**URL:**', '').trim();
                    const linkMatch = urlStr.match(/(https?:\/\/[^\s]+)/);
                    if (linkMatch) {
                        project.urlLink = linkMatch[1];
                        project.urlText = 'Live Site';
                    }
                    inDescription = false;
                } else if (line.startsWith('**Course:**')) {
                    // Ignore or add if needed
                    inDescription = false;
                } else if (line.startsWith('<table') || line.startsWith('</table') || line.startsWith('<tr') || line.startsWith('<td') || line.startsWith('<img')) {
                    // Skip html tables for now
                    inDescription = false;
                } else {
                    if (inDescription && !line.startsWith('**')) {
                        project.description.push(parseInlineMarkdown(line));
                    }
                }
            }

            // Clean description
            project.description = project.description.join('<br><br>');
            
            // Only add if it's a valid project (has a title)
            if (project.title) {
                parsedProjects.push(project);
            }
        }

        return parsedProjects;
    }

    // Parses the **Craft:** value into ordered eras.
    // Format: "<glyph> <tools>", where methods used together in the same period
    // are joined by "+" and successive periods are separated by "→". Tools are
    // optional (unassisted work has none) and belong to the era as a whole.
    // The glyph carries the method; its name lives in the README's "On Method"
    // section rather than on every card.
    const CRAFT_MODES = {
        '✍️': 'handwritten',
        '⌨️': 'autocomplete',
        '💬': 'chat',
        '👾': 'agentic'
    };

    const GLYPH_RE = /^(\p{Extended_Pictographic}️?)\s*/u;

    function parseCraft(str) {
        return str.split('→').map(era => {
            const methods = [];
            let tools = '';

            era.split('+').forEach(part => {
                const text = part.trim();
                const glyphMatch = text.match(GLYPH_RE);
                if (!glyphMatch) return;
                const glyph = glyphMatch[1];
                methods.push({ glyph, mode: CRAFT_MODES[glyph] || '' });
                // Whichever method carries the trailing text holds the era's tools.
                const rest = text.slice(glyphMatch[0].length).trim();
                if (rest) tools = tools ? `${tools}, ${rest}` : rest;
            });

            return methods.length ? { methods, tools } : null;
        }).filter(Boolean);
    }

    // Helper to parse basic inline markdown like bold, italics, and links
    function parseInlineMarkdown(text) {
        return text
            // Parse links: [text](url)
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            // Parse bold: **text**
            .replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>')
            // Parse italics: *text*
            .replace(/\*([^\*]+)\*/g, '<em>$1</em>');
    }

    function extractTags() {
        projects.forEach(p => {
            p.themes.forEach(t => allTags.add(t));
            p.skills.forEach(s => allTags.add(s));
        });
    }

    function renderSelectedTags() {
        selectedTagsContainer.innerHTML = Array.from(activeTags).map(tag => `
            <span class="tag">
                ${tag} <span class="tag-remove" data-tag="${tag}">&times;</span>
            </span>
        `).join('');

        // Add remove listeners
        document.querySelectorAll('.tag-remove').forEach(el => {
            el.addEventListener('click', (e) => {
                const tag = e.target.getAttribute('data-tag');
                activeTags.delete(tag);
                renderSelectedTags();
                renderProjects();
                updateSuggestions(tagSearchInput.value); // refresh dropdown if open
            });
        });
    }

    function updateSuggestions(query) {
        query = query.toLowerCase();
        tagSuggestions.innerHTML = '';
        
        if (!query) {
            tagSuggestions.style.display = 'none';
            return;
        }

        const availableTags = Array.from(allTags)
            .filter(t => !activeTags.has(t) && t.toLowerCase().includes(query))
            .sort();

        if (availableTags.length === 0) {
            tagSuggestions.style.display = 'none';
            return;
        }

        tagSuggestions.innerHTML = availableTags.map(tag => `
            <div class="tag-suggestion-item" data-tag="${tag}">${tag}</div>
        `).join('');

        tagSuggestions.style.display = 'block';

        // Add click listeners
        document.querySelectorAll('.tag-suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                const tag = item.getAttribute('data-tag');
                activeTags.add(tag);
                tagSearchInput.value = '';
                tagSuggestions.style.display = 'none';
                renderSelectedTags();
                renderProjects();
                tagSearchInput.focus();
            });
        });
    }

    function closeAllCraftDetails() {
        document.querySelectorAll('.craft-toggle[aria-expanded="true"]').forEach(btn => {
            btn.setAttribute('aria-expanded', 'false');
            btn.nextElementSibling.hidden = true;
        });
    }

    function renderProjects() {
        grid.innerHTML = '';
        
        const filteredProjects = projects.filter(p => {
            // Search filter
            const matchesSearch = searchQuery === '' || 
                p.title.toLowerCase().includes(searchQuery) || 
                p.description.toLowerCase().includes(searchQuery) ||
                p.themes.some(t => t.toLowerCase().includes(searchQuery)) ||
                p.skills.some(s => s.toLowerCase().includes(searchQuery));

            // Tag filter
            const projectTags = [...p.themes, ...p.skills];
            const matchesTags = activeTags.size === 0 || 
                Array.from(activeTags).every(activeTag => projectTags.includes(activeTag));

            return matchesSearch && matchesTags;
        });

        if (filteredProjects.length === 0) {
            grid.innerHTML = `<div class="loading">No spells found matching those runes.</div>`;
            return;
        }

        filteredProjects.forEach(p => {
            const allProjTags = [...p.themes, ...p.skills];
            
            let repoHtml = '';
            if (p.repoLink) {
                repoHtml = `<a href="${p.repoLink}" target="_blank" rel="noopener noreferrer">
                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                    Source
                </a>`;
            } else if (p.repoText) {
                repoHtml = `<span style="font-size: 0.9rem; color: var(--text-secondary);">${p.repoText}</span>`;
            }

            let urlHtml = '';
            if (p.urlLink) {
                urlHtml = `<a href="${p.urlLink}" target="_blank" rel="noopener noreferrer">
                    <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                    Visit
                </a>`;
            }

            let craftHtml = '';
            if (p.craft.length) {
                // The card shows glyphs alone; the method names live in the detail.
                const label = p.craft
                    .map(e => e.methods.map(m => m.glyph).join(' + '))
                    .join(' → ');
                const ariaLabel = 'Craft: ' + p.craft.map(e => {
                    const names = e.methods.map(m => m.mode).join(' and ');
                    return e.tools ? `${names}, using ${e.tools}` : names;
                }).join(', then ');
                const detailRows = p.craft.map(e => `
                    <div class="craft-era">
                        <span class="craft-era-mode">${e.methods.map(m => `${m.glyph} ${m.mode}`).join(' + ')}</span>
                        ${e.tools ? `<span class="craft-era-tools">${e.tools}</span>` : ''}
                    </div>
                `).join('');

                craftHtml = `
                    <div class="craft-mark">
                        <button class="craft-toggle" type="button"
                                aria-expanded="false" aria-label="${ariaLabel}">${label}</button>
                        <div class="craft-detail" hidden>${detailRows}</div>
                    </div>
                `;
            }

            const cardHtml = `
                <div class="spell-card">
                    <div class="spell-title">${p.title}</div>
                    ${p.meta ? `<div class="spell-meta">${p.meta}</div>` : ''}
                    ${p.quote ? `<div class="spell-quote">${p.quote}</div>` : ''}
                    <div class="spell-desc">${p.description}</div>
                    <button class="read-more-btn">Read more</button>
                    
                    <div class="spell-footer" style="margin-top: auto;">
                        ${craftHtml}
                        <div class="spell-card-tags">
                            ${allProjTags.map(t => `<span class="spell-card-tag">${t}</span>`).join('')}
                        </div>
                        <div class="spell-links">
                            ${urlHtml}
                            ${repoHtml}
                        </div>
                    </div>
                </div>
            `;
            grid.insertAdjacentHTML('beforeend', cardHtml);

            // Handle read more logic
            const currentCard = grid.lastElementChild;
            const descEl = currentCard.querySelector('.spell-desc');
            const btnEl = currentCard.querySelector('.read-more-btn');

            // If text is not clamped (short text), hide the button
            if (descEl.scrollHeight <= descEl.clientHeight) {
                btnEl.style.display = 'none';
            } else {
                btnEl.addEventListener('click', () => {
                    if (descEl.classList.contains('expanded')) {
                        descEl.classList.remove('expanded');
                        btnEl.textContent = 'Read more';
                    } else {
                        descEl.classList.add('expanded');
                        btnEl.textContent = 'Read less';
                    }
                });
            }
        });

        // Craft marks: click/tap toggles the detail open (hover is a pointer-device bonus, CSS-only)
        document.querySelectorAll('.craft-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const isOpen = btn.getAttribute('aria-expanded') === 'true';
                closeAllCraftDetails();
                if (!isOpen) {
                    btn.setAttribute('aria-expanded', 'true');
                    btn.nextElementSibling.hidden = false;
                }
            });
        });

        // Make tags inside cards clickable to filter
        document.querySelectorAll('.spell-card-tag').forEach(tagEl => {
            tagEl.addEventListener('click', (e) => {
                const tagText = e.target.textContent.trim();
                if (!activeTags.has(tagText)) {
                    activeTags.add(tagText);
                    renderSelectedTags();
                    renderProjects();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        });
    }

    // Search logic
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value.toLowerCase();
        renderProjects();
    });

    tagSearchInput.addEventListener('input', (e) => {
        updateSuggestions(e.target.value);
    });

    tagSearchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && tagSearchInput.value === '' && activeTags.size > 0) {
            // Remove the last active tag
            const lastTag = Array.from(activeTags).pop();
            activeTags.delete(lastTag);
            renderSelectedTags();
            renderProjects();
        }
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.tag-input-wrapper')) {
            tagSuggestions.style.display = 'none';
        }
        if (!e.target.closest('.craft-mark')) {
            closeAllCraftDetails();
        }
    });

    // Escape closes an open craft detail
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeAllCraftDetails();
    });

    tagSearchInput.addEventListener('focus', () => {
        updateSuggestions(tagSearchInput.value);
    });

    // Init
    loadSpells();
});
