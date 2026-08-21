// Put this in a file under javascript/ in the webui root — auto-loaded.
(function () {
    const SCOPE_SEL = '#script_txt2img_colorcraft_accordion, #script_img2img_colorcraft_accordion';
    const INPUT_SEL = '.gradio-slider input[type=number]';

    // Matches a complete, valid JS/Python float literal: optional sign,
    // digits (with optional fraction, or a leading-dot fraction), optional
    // exponent. Rejects "+", "-", "e", "1e", "+-", "1-", "", etc.
    const FLOAT_RE = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

    function inScope(el) {
        return el.matches && el.matches(INPUT_SEL) && el.closest(SCOPE_SEL) !== null;
    }

    function sanitize(el) {
        const v = el.value.trim();
        if (!FLOAT_RE.test(v)) {
            el.value = '0';
            // Gradio's Svelte state is keyed off real input events, not
            // direct .value writes — same reasoning as the earlier
            // data-attribute fix, just for value instead of textContent.
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    function intercept(e) {
        const el = e.target;
        if (inScope(el)) {
            e.stopImmediatePropagation();
            sanitize(el);
        }
    }

    document.addEventListener('blur', intercept, true);
    document.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return;
        const el = e.target;
        if (!inScope(el)) return;

        sanitize(el); // always fix a broken value first, regardless of modifier keys

        if (!e.ctrlKey && !e.metaKey) {
            e.stopImmediatePropagation(); // plain Enter: still suppress Gradio's own clamp
        }
        // Ctrl/Cmd+Enter: sanitized, then passes through untouched to the
        // generate-shortcut listener, same as before.
    }, true);

    /*
    document.addEventListener('mousedown', function (e) {
        const el = e.target;
        if (!inScope(el)) return;
        // Defer so the browser has placed the cursor before we select.
        setTimeout(() => el.select(), 0);
    }, true);
    */

    function stripAttrs() {
        document.querySelectorAll(SCOPE_SEL).forEach(scope => {
            scope.querySelectorAll(INPUT_SEL).forEach(el => {
                el.removeAttribute('min');
                el.removeAttribute('max');
            });
        });
    }

    onUiLoaded(function () {
        stripAttrs();
        new MutationObserver(stripAttrs).observe(gradioApp(), { childList: true, subtree: true });
    });
})();
