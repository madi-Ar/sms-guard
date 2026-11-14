/* Результаты API выводятся безопасными DOM-методами. */
(() => {
    "use strict";

    const {api, busy, showMessage} = window.GuardUI;
    const categories = {
        SAFE: ["✓", "Безопасное", "risk-safe"],
        SUSPICIOUS: ["!", "Подозрительное", "risk-suspicious"],
        FRAUDULENT: ["×", "Мошенническое", "risk-dangerous"],
        DANGEROUS: ["×", "Опасная ссылка", "risk-dangerous"],
    };
    const levels = {LOW: "Низкий", MEDIUM: "Средний", HIGH: "Высокий"};
    const languages = {en: "Английский", ru: "Русский", kk: "Казахский", und: "Не определён", other: "Другой"};
    const groups = {
        URGENCY: "Срочность", MONEY_TRANSFER: "Деньги и переводы", BANK_CARD: "Банк и карта",
        ACCOUNT_BLOCK: "Блокировка аккаунта", PRIZE: "Выигрыш и приз", CODE_REQUEST: "Запрос кода или пароля",
    };
    const urlFeatures = {
        is_ip: "IP-адрес вместо домена", has_nonstandard_port: "Нестандартный порт",
        has_at: "Символ @", has_double_slash_path: "Двойной слеш в пути",
        is_shortener: "Сокращатель ссылок", suspicious_keyword_count: "Подозрительные слова",
        has_punycode: "Punycode", has_mixed_scripts: "Смешение алфавитов",
        is_brand_similar: "Сходство с брендом", has_suspicious_tld: "Потенциально рискованная доменная зона",
    };
    const examples = {
        safe: "Встречаемся завтра у библиотеки. Возьми, пожалуйста, синий блокнот.",
        suspicious: "Доставка задержана. Проверьте статус отправления на https://delivery-status.example.",
        fraud: "Срочно! Ваша карта заблокирована. Пришлите код подтверждения и данные карты на https://card-check.example.",
    };

    function element(tag, text = "", className = "") {
        const item = document.createElement(tag);
        item.textContent = text;
        if (className) item.className = className;
        return item;
    }

    function probability(value) {
        return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
            ? value.toLocaleString("ru-RU", {style: "percent", maximumFractionDigits: 2})
            : "Недоступна";
    }

    function categoryBadge(code) {
        const [icon, title, color] = categories[code];
        return element("span", icon + " " + title + " (" + code + ")", "category " + color);
    }

    function textList(parent, title, values, emptyText) {
        parent.append(element("h3", title));
        if (!Array.isArray(values) || !values.length) {
            parent.append(element("p", emptyText));
            return;
        }
        const list = element("ul");
        for (const value of values) list.append(element("li", String(value)));
        parent.append(list);
    }

    function displayUrl(parent, value) {
        parent.append(element("code", String(value), "url-text"));
        if (typeof value !== "string" || value.includes("\u2022")) return;
        try {
            const parsed = new URL(/^https?:\/\//i.test(value) ? value : "http://" + value);
            parent.append(element("strong", "Домен: " + parsed.hostname, "url-text"));
        } catch {
            // Safe DOM rendering.
        }
    }

    function renderUrl(parent, data) {
        const article = element("article", "", "url-item");
        if (categories[data.category]) article.append(categoryBadge(data.category));
        displayUrl(article, data.url);
        article.append(element("p", "Вероятность URL-модели (P_url): " + probability(data.p_url)));
        const featureNames = Object.entries(urlFeatures)
            .filter(([key]) => Number(data.features?.[key]) > 0).map(([, title]) => title);
        if (data.features?.is_https === 0) featureNames.push("Не используется HTTPS");
        textList(article, "Признаки ссылки", featureNames, "Перечисленные признаки не обнаружены.");
        textList(article, "Объяснение оценки ссылки", data.explanation, "Объяснение не передано сервером.");
        parent.append(article);
    }

    function renderResult(data, source = "", includeLink = true) {
        if (!categories[data.category] || !["sms", "url"].includes(data.type)) {
            throw new Error("Получен неожиданный результат анализа. Повторите запрос позже.");
        }
        const target = document.querySelector("#result");
        target.replaceChildren();
        const heading = element("div", "", "result-heading");
        heading.append(element("h2", "Результат проверки", "h4 mb-0"), categoryBadge(data.category));
        target.append(heading);
        if (typeof data.masked_input === "string" || source) {
            target.append(element("h3", typeof data.masked_input === "string" ? "Маскированный ввод" : "Проверенный текст"));
            target.append(element("p", data.masked_input ?? source, "plain-input"));
        }
        if (data.created_at) {
            const date = new Date(data.created_at);
            target.append(element("p", "Дата проверки: " + (Number.isNaN(date.getTime()) ? "не передана" : date.toLocaleString("ru-RU")), "small text-secondary mt-3"));
        }
        if (data.type === "sms") {
            target.append(element("p", "Уровень риска: " + (levels[data.risk_level] || "Не определён"), "fw-semibold mt-3"));
            if (typeof data.risk_score === "number" && Number.isFinite(data.risk_score)
                && data.risk_score >= 0 && data.risk_score <= 100) {
                target.append(element("p", "Оценка риска (Risk Score): " + data.risk_score + "/100", "fw-semibold"));
                const track = element("div", "", "risk-track " + categories[data.category][2]);
                track.setAttribute("role", "progressbar");
                track.setAttribute("aria-label", "Оценка риска");
                track.setAttribute("aria-valuemin", "0");
                track.setAttribute("aria-valuemax", "100");
                track.setAttribute("aria-valuenow", String(data.risk_score));
                const fill = element("div", "", "risk-fill");
                fill.style.width = data.risk_score + "%";
                track.append(fill);
                target.append(track);
            }
            const metrics = element("dl", "", "metric-grid");
            for (const [label, value] of [
                ["SMS-модель (P_sms)", probability(data.p_sms)],
                ["Ссылки (P_url_max)", data.p_url_max === null ? "Ссылки не обнаружены" : probability(data.p_url_max)],
                ["Язык сообщения", languages[data.language] || "Не определён"],
            ]) {
                const item = element("div");
                item.append(element("dt", label), element("dd", value));
                metrics.append(item);
            }
            target.append(metrics);
            const matchedGroups = data.features?.social_engineering_groups;
            textList(target, "Группы социальной инженерии",
                Array.isArray(matchedGroups) ? matchedGroups.map(code => groups[code] || code) : [],
                "Группы не обнаружены.");
            textList(target, "Объяснение результата", data.explanation, "Объяснение не передано сервером.");
            target.append(element("h3", "Рекомендация"), element("p", data.recommendation || "Рекомендация не передана сервером."));
            if (Array.isArray(data.urls) && data.urls.length) {
                target.append(element("h3", "Найденные ссылки"));
                for (const url of data.urls) renderUrl(target, url);
            }
        } else {
            if (data.risk_score === null) target.append(element("p", "Итоговый Risk Score для отдельной ссылки не рассчитывается.", "small text-secondary mt-3"));
            renderUrl(target, data);
        }
        if (data.disclaimer) target.append(element("p", data.disclaimer, "notice mt-4"));
        if (includeLink && Number.isSafeInteger(data.id) && data.id > 0) {
            const link = element("a", "Открыть сохранённый результат", "btn btn-outline-primary mt-2");
            // Safe DOM rendering.
            link.setAttribute("href", "/checks/" + data.id + "/");
            target.append(link);
        }
        target.hidden = false;
    }

    const form = document.querySelector("#check-form");
    if (form) {
        const input = document.querySelector("#check-input");
        const error = document.querySelector("#input-error");
        const counter = document.querySelector("#counter");
        const result = document.querySelector("#result");
        const clear = document.querySelector("#clear-button");

        function updateInput() {
            if (counter) counter.textContent = Array.from(input.value).length + "/1000";
            showMessage(error, "");
            input.removeAttribute("aria-invalid");
        }
        input.addEventListener("input", updateInput);
        clear.addEventListener("click", () => {
            if (form.getAttribute("aria-busy") === "true") return;
            input.value = "";
            updateInput();
            result.replaceChildren();
            result.hidden = true;
            input.focus();
        });
        for (const button of document.querySelectorAll("[data-example]")) {
            button.addEventListener("click", () => {
                if (form.getAttribute("aria-busy") === "true") return;
                input.value = examples[button.dataset.example];
                updateInput();
                result.hidden = true;
                input.focus();
            });
        }
        form.addEventListener("submit", async event => {
            event.preventDefault();
            if (form.getAttribute("aria-busy") === "true") return;
            updateInput();
            const source = input.value;
            const limit = form.dataset.kind === "sms" ? 1000 : 2048;
            if (!source.trim() || Array.from(source).length > limit) {
                showMessage(error, !source.trim() ? "Введите текст для проверки." : "Допустимо не более " + limit + " символов.");
                input.setAttribute("aria-invalid", "true");
                input.focus();
                return;
            }
            const key = form.dataset.kind === "sms" ? "text" : "url";
            busy(form, true);
            clear.disabled = true;
            result.hidden = true;
            try {
                const data = await api(form.dataset.endpoint, "POST", {[key]: source});
                renderResult(data, source);
                result.focus();
            } catch (failure) {
                showMessage(error, failure.message);
                input.setAttribute("aria-invalid", "true");
            } finally {
                busy(form, false);
                clear.disabled = false;
            }
        });
    }

    const saved = document.querySelector("#saved-check");
    if (saved) {
        const retry = document.querySelector("#retry-result");
        async function loadResult() {
            if (saved.getAttribute("aria-busy") === "true") return;
            saved.setAttribute("aria-busy", "true");
            saved.querySelector(".loading").hidden = false;
            retry.hidden = true;
            showMessage(document.querySelector("#detail-error"), "");
            try {
                renderResult(await api(saved.dataset.endpoint), "", false);
            } catch (error) {
                showMessage(document.querySelector("#detail-error"), error.message);
                retry.hidden = false;
            } finally {
                saved.querySelector(".loading").hidden = true;
                saved.setAttribute("aria-busy", "false");
            }
        }
        retry.addEventListener("click", loadResult);
        loadResult();
    }
})();