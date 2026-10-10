(() => {
  const sidebar = document.getElementById('sidebar');
  const menuToggle = document.querySelector('.menu-toggle');
  const search = document.getElementById('site-search');
  const toast = document.getElementById('toast');
  const sections = [...document.querySelectorAll('main > section[data-searchable]')];
  const cards = [...document.querySelectorAll('.step-card, .feature-card, .cleanup-card, .roadmap-grid article')];
  let emptyState = null;
  let toastTimer = 0;

  const showToast = (message) => {
    toast.textContent = message;
    toast.classList.add('show');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove('show'), 1900);
  };

  const closeMenu = () => {
    sidebar.classList.remove('open');
    menuToggle.setAttribute('aria-expanded', 'false');
  };

  menuToggle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    menuToggle.setAttribute('aria-expanded', String(open));
  });

  sidebar.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));

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
      emptyState.textContent = `No public guide matches “${search.value.trim()}”. Try “export”, “hair”, “logo”, or “shortcut”.`;
      document.getElementById('main-content').append(emptyState);
    }
    updateActiveLink();
  };

  search.addEventListener('input', setSearch);
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

  document.querySelectorAll('.key[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      const value = button.dataset.copy || '';
      try {
        await navigator.clipboard.writeText(value);
        showToast(`Copied ${value}`);
      } catch {
        const area = document.createElement('textarea');
        area.value = value;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.append(area);
        area.select();
        document.execCommand('copy');
        area.remove();
        showToast(`Copied ${value}`);
      }
    });
  });

  const links = [...document.querySelectorAll('.side-link')];
  const byId = new Map(links.map((link) => [link.getAttribute('href').slice(1), link]));
  const readingLine = () => window.scrollY + Math.min(220, window.innerHeight * 0.3);
  const updateActiveLink = () => {
    const visibleSections = sections.filter((section) => !section.hidden);
    if (!visibleSections.length) return;
    const line = readingLine();
    let current = visibleSections[0];
    visibleSections.forEach((section) => {
      const top = section.getBoundingClientRect().top + window.scrollY;
      if (top <= line + 24) current = section;
    });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 24) {
      current = visibleSections[visibleSections.length - 1];
    }
    links.forEach((link) => link.classList.toggle('active', link === byId.get(current.id)));
  };
  let scrollFrame = 0;
  const onScroll = () => {
    if (scrollFrame) return;
    scrollFrame = window.requestAnimationFrame(() => {
      scrollFrame = 0;
      updateActiveLink();
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  window.addEventListener('hashchange', updateActiveLink);
  updateActiveLink();

  const hoverTargets = [...document.querySelectorAll('.step-card, .feature-card, .cleanup-card, .edition-card, .roadmap-grid article')];
  hoverTargets.forEach((target, index) => {
    target.classList.add('hover-surface');
    target.style.transitionDelay = `${(index % 4) * 35}ms`;
    target.addEventListener('mousemove', (event) => {
      const rect = target.getBoundingClientRect();
      target.style.setProperty('--mx', `${event.clientX - rect.left}px`);
      target.style.setProperty('--my', `${event.clientY - rect.top}px`);
    });
  });

  const hero = document.querySelector('.hero');
  const characterSlot = document.querySelector('.character-slot');
  if (hero && characterSlot) {
    hero.addEventListener('mousemove', (event) => {
      const rect = hero.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      const y = (event.clientY - rect.top) / rect.height - 0.5;
      characterSlot.style.setProperty('--tilt-y', `${x * 12}deg`);
      characterSlot.style.setProperty('--tilt-x', `${-y * 12}deg`);
    });
    hero.addEventListener('mouseleave', () => {
      characterSlot.style.setProperty('--tilt-y', '0deg');
      characterSlot.style.setProperty('--tilt-x', '0deg');
    });
  }

  const revealTargets = [...document.querySelectorAll('.step-card, .feature-card, .cleanup-card, .workflow-row, .roadmap-grid article')];
  if ('IntersectionObserver' in window) {
    revealTargets.forEach((target) => target.classList.add('reveal'));
    const revealObserver = new IntersectionObserver((entries, current) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          current.unobserve(entry.target);
        }
      });
    }, { threshold: .08 });
    revealTargets.forEach((target) => revealObserver.observe(target));
    window.setTimeout(() => revealTargets.forEach((target) => target.classList.add('revealed')), 1500);
  }
})();
