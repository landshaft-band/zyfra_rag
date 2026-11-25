const form = document.querySelector("#composer");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-question");
const resetButton = document.querySelector("#reset-chat");
const examples = document.querySelector("#examples");
const chatBody = document.querySelector("#chat-body");
const chatLog = document.querySelector("#chat-log");
const sourcesList = document.querySelector("#sources-list");
const sourceCount = document.querySelector("#source-count");
const connection = document.querySelector("#connection");
const historyControl = document.querySelector("#history-control");
const conversationSelect = document.querySelector("#conversation-select");
const historyNotice = document.querySelector("#history-notice");
const sourcesHint = sourcesList.textContent.trim();

const statusTitles = {
    answer: "Черновик ответа",
    clarify: "Нужно уточнить",
    no_answer: "Недостаточно сведений",
    search_only: "Найдены источники",
    error: "Ошибка запроса",
};

let busy = false;
let useHistory = true;
let conversationId = null;
let conversations = [];
let pendingSubmission = null;

const dateFormat = new Intl.DateTimeFormat("ru", {
    day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
});

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function setBusy(value, answering = false) {
    busy = value;
    sendButton.disabled = value;
    sendButton.textContent = answering ? "Готовим ответ…" : "Отправить";
    resetButton.disabled = value || (!conversationId && !chatLog.childElementCount);
    conversationSelect.disabled = value;
    questionInput.readOnly = value;
}

function scrollChat() {
    chatBody.scrollTop = chatBody.scrollHeight;
}

function renderSources(sources, selectedId) {
    sourcesList.replaceChildren();
    sourceCount.hidden = sources.length === 0;
    sourceCount.textContent = String(sources.length);
    if (!sources.length) {
        sourcesList.append(element("p", "empty-sources", sourcesHint));
        return null;
    }

    let active = null;
    for (const source of sources) {
        const card = element("details", "source-card");
        const summary = element("summary");
        summary.append(element("span", "source-id", source.source_id));
        summary.append(document.createTextNode(source.title));
        if (source.source_id === selectedId) {
            card.open = true;
            active = summary;
        }

        const detail = element("div", "source-detail");
        const file = source.start_line
            ? `${source.file_name} · строка ${source.start_line}`
            : source.file_name;
        detail.append(
            element("p", "source-locator", source.locator.join(" · ")),
            element("p", "source-file", file),
            element("p", "source-excerpt", source.text),
        );
        card.append(summary, detail);
        sourcesList.append(card);
    }
    return active;
}

function responseMessage(result) {
    const box = element("article", "message message-assistant");
    box.dataset.status = result.status;
    box.append(element("h2", "message-header", statusTitles[result.status]));

    if (result.message) box.append(element("p", null, result.message));
    if (result.items?.length) {
        const items = element("ol", "answer-items");
        for (const item of result.items) {
            const line = element("li", null, item.text);
            for (const id of item.source_ids) {
                const citation = element("button", "citation", `[${id}]`);
                citation.type = "button";
                citation.setAttribute("aria-label", `Показать источник ${id}`);
                citation.addEventListener("click", () => {
                    const summary = renderSources(result.sources, id);
                    summary?.focus({preventScroll: true});
                    summary?.scrollIntoView({block: "nearest"});
                });
                line.append(citation);
            }
            items.append(line);
        }
        box.append(items);
    }
    if (result.missing_information?.length) {
        const details = element("ul");
        for (const detail of result.missing_information) details.append(element("li", null, detail));
        box.append(details);
    }
    return box;
}

async function requestJSON(path, options) {
    const response = await fetch(path, options);
    const result = await response.json().catch(() => null);
    if (!response.ok) {
        const error = new Error(typeof result?.detail === "string" ? result.detail : "Запрос к серверу не выполнен.");
        error.status = response.status;
        error.historyDisabled = response.headers.get("X-Rag-History") === "disabled";
        throw error;
    }
    return result;
}

function setHistoryNotice(text = "") {
    historyNotice.textContent = text;
    historyNotice.hidden = !text;
}

function selectConversation(id) {
    conversationId = id;
    const url = new URL(window.location.href);
    if (useHistory) url.searchParams.set("chat", id || "new");
    window.history.replaceState(null, "", url);
    conversationSelect.replaceChildren(new Option("Новый чат", ""));
    for (const conversation of conversations) {
        conversationSelect.append(new Option(dateFormat.format(new Date(conversation.created_at)), conversation.id));
    }
    conversationSelect.value = id || "";
}

function userMessage(text) {
    const user = element("article", "message message-user");
    user.append(element("p", null, text));
    return user;
}

async function openConversation(id) {
    const {messages} = await requestJSON(`/api/conversations/${encodeURIComponent(id)}`);
    selectConversation(id);
    chatLog.replaceChildren();
    let sources = [];
    for (const message of messages) {
        if (message.role === "user") {
            chatLog.append(userMessage(message.content));
        } else if (message.result) {
            chatLog.append(responseMessage(message.result));
            sources = message.result.sources;
        } else {
            chatLog.append(element("p", "message message-assistant", message.content));
        }
    }
    examples.hidden = messages.length > 0;
    renderSources(sources, sources[0]?.source_id);
    scrollChat();
}

function newConversation() {
    selectConversation(null);
    chatLog.replaceChildren();
    examples.hidden = false;
    renderSources([]);
}

const queuedText = "Вопрос в очереди…";
const runningText = "Ищем сведения и готовим ответ…";
const queuedNotice = "Вопрос сохранён в очереди. Обновите страницу или снова откройте диалог, чтобы проверить статус.";

async function waitForJob(job, showStatus) {
    const deadline = Date.now() + 120000;
    while (job.status === "queued" || job.status === "running") {
        showStatus(job.status === "running" ? runningText : queuedText);
        await new Promise((resolve) => setTimeout(resolve, 1000));
        try {
            job = await requestJSON(`/api/jobs/${encodeURIComponent(job.id)}`, {signal: AbortSignal.timeout(5000)});
        } catch (error) {
            if (error.status && error.status < 500) throw error;
        }
        if (Date.now() >= deadline && (job.status === "queued" || job.status === "running")) {
            throw new Error(queuedNotice);
        }
    }
    return job;
}

async function reopenConversation(id) {
    // Read active jobs first so a completion during reload can still be found in the history snapshot.
    const active = await requestJSON(`/api/conversations/${encodeURIComponent(id)}/jobs`);
    await openConversation(id);
    setHistoryNotice();
    if (!active.length) return;
    let failure = "";
    try {
        for (const job of active) {
            const finalJob = await waitForJob(job, setHistoryNotice);
            if (finalJob.status === "failed") failure = finalJob.error;
        }
        await openConversation(id);
        setHistoryNotice(failure);
    } catch {
        setHistoryNotice(queuedNotice);
    }
}

async function loadHistory() {
    let listed = false;
    setBusy(true);
    try {
        conversations = await requestJSON("/api/conversations");
        listed = true;
        historyControl.hidden = false;
        const selected = new URL(window.location.href).searchParams.get("chat");
        if (selected === "new") {
            selectConversation(null);
        } else {
            const target = conversations.find((chat) => chat.id === selected) ?? conversations[0];
            if (target) await reopenConversation(target.id);
            else selectConversation(null);
        }
    } catch (error) {
        useHistory = listed || (!error.historyDisabled && error.status !== 404);
        setHistoryNotice(useHistory ? "Не удалось загрузить историю. Обновите страницу." : "История диалога отключена.");
    } finally {
        setBusy(false);
    }
}

function postJSON(path, data) {
    return requestJSON(path, {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data),
    });
}

async function ask(question, showStatus) {
    if (!useHistory) return postJSON("/api/ask", {question});
    if (!conversationId) {
        const created = await requestJSON("/api/conversations", {method: "POST"});
        conversations.unshift(created);
        selectConversation(created.id);
        historyControl.hidden = false;
    }
    const sameRequest = pendingSubmission?.conversationId === conversationId && pendingSubmission.question === question;
    if (!sameRequest) pendingSubmission = {conversationId, question, id: crypto.randomUUID()};
    const path = `/api/conversations/${encodeURIComponent(conversationId)}/jobs`;
    let job;
    try {
        job = await postJSON(path, {question, request_id: pendingSubmission.id});
    } catch (error) {
        if (error.status && error.status < 500) pendingSubmission = null;
        throw error;
    }
    pendingSubmission = null;
    try {
        const finalJob = await waitForJob(job, showStatus);
        if (finalJob.status !== "completed") {
            throw new Error(finalJob.error || "Не удалось обработать задание.");
        }
        return finalJob.result;
    } catch (error) {
        error.jobAccepted = true;
        throw error;
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (busy || !question) return;

    examples.hidden = true;
    questionInput.value = "";
    const user = userMessage(question);
    const pending = element("p", "message message-assistant", "Ищем сведения в базе знаний…");
    pending.setAttribute("role", "status");
    chatLog.append(user, pending);
    setBusy(true, true);
    setHistoryNotice();
    scrollChat();

    try {
        const result = await ask(question, (status) => { pending.textContent = status; });
        pending.replaceWith(responseMessage(result));
        renderSources(result.sources, result.sources[0]?.source_id);
        if (useHistory) {
            try {
                await openConversation(conversationId);
            } catch {
                setHistoryNotice("Ответ сохранён. Обновите страницу, чтобы загрузить историю.");
            }
        }
    } catch (error) {
        const message = error instanceof TypeError
            ? "Связь с сервером прервалась. Попробуйте ещё раз."
            : error.message;
        if (useHistory && conversationId) {
            try {
                await openConversation(conversationId);
            } catch {
                user.remove();
            }
        }
        if (!pending.isConnected) chatLog.append(pending);
        pending.replaceWith(responseMessage({status: "error", message}));
        examples.hidden = true;
        if (!error.jobAccepted) questionInput.value = question;
    } finally {
        setBusy(false);
        scrollChat();
        questionInput.focus({preventScroll: true});
    }
});

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!busy) form.requestSubmit();
    }
});

examples.querySelectorAll("[data-question]").forEach((example) => {
    example.addEventListener("click", () => {
        questionInput.value = example.dataset.question;
        questionInput.focus();
    });
});

resetButton.addEventListener("click", () => {
    newConversation();
    questionInput.value = "";
    setHistoryNotice();
    setBusy(false);
    questionInput.focus();
});

conversationSelect.addEventListener("change", async () => {
    setBusy(true);
    try {
        const selected = conversationSelect.value;
        if (selected) await reopenConversation(selected);
        else {
            newConversation();
            setHistoryNotice();
        }
        questionInput.value = "";
    } catch (error) {
        conversationSelect.value = conversationId || "";
        setHistoryNotice(error.message);
    } finally {
        setBusy(false);
    }
});

async function checkHealth() {
    let ready = false;
    try {
        const response = await fetch("/api/health", {cache: "no-store"});
        ready = response.ok && (await response.json()).status === "ready";
    } catch {
        ready = false;
    }
    connection.dataset.state = ready ? "ready" : "offline";
    connection.textContent = ready ? "База знаний подключена" : "База знаний недоступна";
}

checkHealth();
loadHistory();
