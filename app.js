(() => {
  const sidebar = document.getElementById('sidebar');
  const menuToggle = document.querySelector('.menu-toggle');
  const search = document.getElementById('site-search');
  const toast = document.getElementById('toast');
  const sections = [...document.querySelectorAll('main > section[data-searchable]')];
  const cards = [...document.querySelectorAll('.step-card, .feature-card, .faq-list details')];
  const sideLinks = [...document.querySelectorAll('.side-link')];
  let emptyState = null;
  let toastTimer = 0;

  /* Toast */
  const showToast = (message) => {
    toast.textContent = message;
    toast.classList.add('show');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove('show'), 1900);
  };

  /* Mobile menu */
  const closeMenu = () => {
    sidebar.classList.remove('open');
    menuToggle.setAttribute('aria-expanded', 'false');
  };
  menuToggle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    menuToggle.setAttribute('aria-expanded', String(open));
  });
  sidebar.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));

  /* Scrollspy */
  const updateActiveLink = () => {
    let current = sections[0];
    for (const section of sections) {
      if (section.hidden) continue;
      if (section.getBoundingClientRect().top <= 120) current = section;
    }
    if (!current) return;
    sideLinks.forEach((link) =>
      link.classList.toggle('active', link.getAttribute('href') === `#${current.id}`)
    );
  };
  document.addEventListener('scroll', updateActiveLink, { passive: true });

  /* Search */
  const setSearch = () => {
    const query = search.value.trim().toLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);
    let visibleSections = 0;

    sections.forEach((section) => {
      const haystack = `${section.dataset.searchable || ''} ${section.textContent}`.toLowerCase();
      const visible = !terms.length || terms.every((term) => haystack.includes(term));
      section.hidden = !visible;
      if (visible) visibleSections += 1;
    });

    cards.forEach((card) => {
      const haystack = `${card.dataset.searchable || ''} ${card.textContent}`.toLowerCase();
      card.hidden = Boolean(terms.length && !terms.every((term) => haystack.includes(term)));
    });

    if (emptyState) emptyState.remove();
    if (query && visibleSections === 0) {
      emptyState = document.createElement('div');
      emptyState.className = 'search-empty';
      emptyState.textContent = `No guide matches “${search.value.trim()}”. Try “export”, “hair”, “matte”, or “shortcut”.`;
      document.getElementById('main-content').append(emptyState);
    }
    updateActiveLink();
  };
  search.addEventListener('input', setSearch);

  /* Keyboard: "/" focuses search, Escape clears */
  document.addEventListener('keydown', (event) => {
    if (event.key === '/' && document.activeElement !== search) {
      event.preventDefault();
      search.focus();
    }
    if (event.key === 'Escape' && document.activeElement === search) {
      search.value = '';
      setSearch();
      search.blur();
    }
  });

  /* Copyable shortcut keys */
  document.querySelectorAll('.key[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      const value = button.dataset.copy;
      try {
        await navigator.clipboard.writeText(value);
        showToast(`Copied “${value}”`);
      } catch {
        showToast(value);
      }
    });
  });

  updateActiveLink();
})();
