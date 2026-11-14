/* Session API transport. */
(() => {
    "use strict";

    const messages = {
        400: "Проверьте введённые данные.",
        401: "Необходимо войти в аккаунт.",
        403: "Доступ запрещён. Обновите страницу и повторите запрос.",
        404: "Запись не найдена или недоступна.",
        429: "Слишком много запросов. Подождите минуту и повторите попытку.",
        503: "Модели анализа временно недоступны. Попробуйте позже.",
    };

    function showMessage(element, message) {
        element.textContent = message;
        element.hidden = !message;
    }

    function busy(form, value) {
        form.setAttribute("aria-busy", String(value));
        form.querySelector('[type="submit"]').disabled = value;
        form.querySelector(".loading").hidden = !value;
    }

    async function api(endpoint, method = "GET", body) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30000);
        try {
            const headers = {Accept: "application/json"};
            if (method !== "GET") {
                headers["X-CSRFToken"] = document.querySelector('[name="csrfmiddlewaretoken"]').value;
                headers["Content-Type"] = "application/json";
            }
            const response = await fetch(endpoint, {
                method, headers, credentials: "same-origin", signal: controller.signal,
                ...(body === undefined ? {} : {body: JSON.stringify(body)}),
            });
            if (response.status === 204) return null;
            let data;
            try {
                data = await response.json();
            } catch {
                throw new Error(messages[response.status] || "Получен неожиданный ответ сервера. Повторите запрос позже.");
            }
            if (!response.ok) {
                // Safe DOM rendering.
                const message = response.status === 401 && endpoint === "/api/v1/auth/login/"
                    ? "Неверные логин или пароль."
                    : response.status === 400 && data?.error?.code === "VALIDATION_ERROR"
                    && typeof data.error.message === "string" ? data.error.message : messages[response.status];
                throw new Error(message || "Сервис временно недоступен. Повторите запрос позже.");
            }
            if (!data || typeof data !== "object" || Array.isArray(data)) {
                throw new Error("Получен неожиданный ответ сервера. Повторите запрос позже.");
            }
            return data;
        } catch (error) {
            if (error.name === "AbortError") {
                throw new Error("Сервер долго не отвечает. Повторите запрос.");
            }
            if (error instanceof TypeError) {
                throw new Error("Не удалось связаться с сервером. Проверьте соединение.");
            }
            throw error;
        } finally {
            clearTimeout(timer);
        }
    }

    window.GuardUI = {api, busy, showMessage};

    const auth = document.querySelector("#auth-form");
    if (auth) {
        auth.addEventListener("submit", async event => {
            event.preventDefault();
            if (auth.getAttribute("aria-busy") === "true") return;
            const error = document.querySelector("#auth-error");
            const success = document.querySelector("#auth-success");
            showMessage(error, "");
            showMessage(success, "");
            const username = auth.elements.username.value;
            const password = auth.elements.password.value;
            const registering = auth.dataset.mode === "register";
            if (!username.trim() || !password) {
                showMessage(error, "Введите логин и пароль.");
                return;
            }
            const body = {username, password};
            if (registering) {
                const email = auth.elements.email;
                if (!email.value.trim() || !email.checkValidity()) {
                    showMessage(error, "Укажите корректную электронную почту.");
                    return;
                }
                if (password !== document.querySelector("#password-confirmation").value) {
                    showMessage(error, "Пароли не совпадают.");
                    return;
                }
                body.email = email.value;
            }
            busy(auth, true);
            try {
                await api(auth.dataset.endpoint, "POST", body);
                if (registering) {
                    auth.reset();
                    showMessage(success, "Аккаунт создан. Теперь войдите с вашим логином и паролем.");
                } else {
                    window.location.assign("/check/sms/");
                }
            } catch (failure) {
                showMessage(error, failure.message);
            } finally {
                busy(auth, false);
            }
        });
    }

    const logout = document.querySelector("#logout-button");
    if (logout) {
        logout.addEventListener("click", async () => {
            if (logout.disabled) return;
            logout.disabled = true;
            try {
                await api(logout.dataset.endpoint, "POST", {});
                window.location.assign("/login/");
            } catch (error) {
                showMessage(document.querySelector("#session-error"), error.message);
            } finally {
                logout.disabled = false;
            }
        });
    }
})();
