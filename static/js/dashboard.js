/* Dashboard uses the existing authenticated API transport and safe DOM nodes. */
(() => {
    "use strict";
    const dashboard = document.querySelector("#dashboard");
    if (!dashboard) return;
    const form = document.querySelector("#period-form");
    const period = document.querySelector("#period");
    const retry = document.querySelector("#dashboard-retry");
    const charts = [];
    const colors = ["#22764c", "#b68013", "#b12f41"];
    const labels = ["Безопасные", "Подозрительные", "Опасные"];
    const keys = ["SAFE", "SUSPICIOUS", "DANGEROUS"];

    function node(tag, text) {
        const value = document.createElement(tag);
        if (text !== undefined) value.textContent = String(text);
        return value;
    }

    function table(id, headings, rows) {
        const target = document.getElementById(id);
        target.replaceChildren();
        const result = node("table");
        result.className = "table align-middle";
        const head = node("thead");
        const heading = node("tr");
        headings.forEach(text => {
            const cell = node("th", text);
            cell.scope = "col";
            heading.append(cell);
        });
        head.append(heading);
        const body = node("tbody");
        rows.forEach(values => {
            const row = node("tr");
            values.forEach(value => {
                const cell = node("td");
                if (value instanceof Node) cell.append(value);
                else cell.textContent = String(value ?? "—");
                row.append(cell);
            });
            body.append(row);
        });
        result.append(head, body);
        target.append(result);
    }

    function draw(id, type, title, chartLabels, datasets, hasData, scales) {
        const canvas = document.getElementById(id);
        canvas.parentElement.hidden = !hasData || typeof window.Chart !== "function";
        if (!hasData || typeof window.Chart !== "function") return;
        charts.push(new window.Chart(canvas, {
            type,
            data: {labels: chartLabels, datasets},
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {title: {display: true, text: title}, legend: {display: true, position: "bottom"}},
                ...(scales ? {scales} : {}),
            },
        }));
    }

    function render(data) {
        charts.splice(0).forEach(chart => chart.destroy());
        const total = data.sms_count + data.url_count;
        for (const key of ["sms_count", "url_count", "safe_count", "suspicious_count", "dangerous_count"]) {
            document.getElementById(key).textContent = String(data[key]);
        }
        document.getElementById("average_risk_score").textContent = Number.isFinite(data.average_risk_score)
            ? data.average_risk_score.toFixed(1) : "—";
        document.getElementById("dashboard-range").textContent =
            `С ${data.date_from} по ${data.date_to} · ${data.timezone} · ${data.period} дней`;
        document.getElementById("dashboard-empty").hidden = total > 0;
        document.getElementById("categories-empty").hidden = total > 0;
        document.getElementById("daily-empty").hidden = total > 0;
        document.getElementById("features-empty").hidden = data.top_features.length > 0;
        document.getElementById("recent-empty").hidden = data.recent_checks.length > 0;
        window.GuardUI.showMessage(document.getElementById("chart-error"), typeof window.Chart === "function"
            ? "" : "Библиотека графиков недоступна. Значения доступны в таблицах.");
        table("categories-text", ["Категория", "Проверок"], data.category_distribution.map(row => [row.label, row.count]));
        table("daily-text", ["Дата", ...labels], data.daily.map(row => [row.date, ...keys.map(key => row[key])]));
        table("features-text", ["Группа", "Проверок с признаком"], data.top_features.map(row => [row.title, row.count]));
        const quantity = {beginAtZero: true, ticks: {precision: 0}, title: {display: true, text: "Количество проверок"}};
        draw("categories-chart", "doughnut", "Распределение категорий", labels,
            [{label: "Проверок", data: data.category_distribution.map(row => row.count), backgroundColor: colors}], total > 0);
        draw("daily-chart", "line", "Проверки по дням", data.daily.map(row => row.date),
            keys.map((key, index) => ({label: labels[index], data: data.daily.map(row => row[key]),
                borderColor: colors[index], backgroundColor: colors[index], tension: 0.1})), total > 0,
            {x: {title: {display: true, text: "Дата"}}, y: quantity});
        draw("features-chart", "bar", "Топ-5 групп социальной инженерии", data.top_features.map(row => row.title),
            [{label: "Проверок с признаком", data: data.top_features.map(row => row.count), backgroundColor: "#425779"}], data.top_features.length > 0,
            {x: {title: {display: true, text: "Группа признаков"}}, y: quantity});
        const date = value => value ? new Date(value).toLocaleString("ru-RU", {timeZone: data.timezone}) : "—";
        table("recent-checks", ["Дата", "Тип", "Категория", "Уровень риска", "Risk Score", "Действие"], data.recent_checks.map(row => {
            let action = "Доступно владельцу";
            // Construct the internal path locally, never use an API-supplied URL as href.
            if (row.detail_url && Number.isSafeInteger(row.id) && row.id > 0) {
                action = node("a", "Подробнее");
                action.setAttribute("href", `/checks/${row.id}/`);
            }
            return [date(row.created_at), row.type.toUpperCase(), row.category, row.risk_level, row.risk_score, action];
        }));
        if (document.getElementById("active-models")) {
            document.getElementById("models-empty").hidden = data.active_models.length > 0;
            table("active-models", ["Тип", "Версия", "Алгоритм", "Сохранённые метрики", "Дата обучения"], data.active_models.map(model => [
                model.model_type.toUpperCase(), model.version, model.algorithm,
                Object.entries(model.metrics).map(([key, value]) => `${key} = ${value.toFixed(4)}`).join(", ") || "—", date(model.trained_at),
            ]));
        }
    }

    async function load() {
        if (form.getAttribute("aria-busy") === "true") return;
        form.setAttribute("aria-busy", "true");
        period.disabled = true;
        form.querySelector("button").disabled = true;
        retry.hidden = true;
        document.getElementById("dashboard-loading").hidden = false;
        document.getElementById("dashboard-data").hidden = true;
        window.GuardUI.showMessage(document.getElementById("dashboard-error"), "");
        try {
            const data = await window.GuardUI.api(`${dashboard.dataset.endpoint}?period=${encodeURIComponent(period.value)}`);
            render(data);
            const url = new URL(window.location.href);
            url.searchParams.set("period", String(data.period));
            window.history.replaceState(null, "", url);
            document.getElementById("dashboard-data").hidden = false;
        } catch (error) {
            window.GuardUI.showMessage(document.getElementById("dashboard-error"), error.message);
            retry.hidden = false;
        } finally {
            form.setAttribute("aria-busy", "false");
            period.disabled = false;
            form.querySelector("button").disabled = false;
            document.getElementById("dashboard-loading").hidden = true;
        }
    }
    form.addEventListener("submit", event => { event.preventDefault(); load(); });
    period.addEventListener("change", load);
    retry.addEventListener("click", load);
    load();
})();
