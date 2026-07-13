import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function chainCallback(target, property, callback) {
    const original = target[property];
    target[property] = function (...args) {
        const originalResult = original?.apply(this, args);
        return callback.apply(this, args) ?? originalResult;
    };
}

app.registerExtension({
    name: "DaSiWa.AutoLong.Preview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "DaSiWaAutoLongStream") return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.display = "none";

            const video = document.createElement("video");
            video.controls = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;
            video.style.display = "block";
            video.style.width = "100%";
            video.style.borderRadius = "6px";
            container.appendChild(video);

            const download = document.createElement("a");
            download.textContent = "下载最终视频";
            download.style.display = "block";
            download.style.margin = "8px 0 2px";
            download.style.padding = "7px 10px";
            download.style.borderRadius = "6px";
            download.style.background = "#2f7d4a";
            download.style.color = "white";
            download.style.textAlign = "center";
            download.style.textDecoration = "none";
            download.style.cursor = "pointer";
            container.appendChild(download);

            const widget = this.addDOMWidget(
                "autolong_video_preview",
                "preview",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                    getValue: () => null,
                    setValue: () => {},
                },
            );

            widget.computeSize = (width) => {
                if (!video.src || container.style.display === "none") {
                    return [width, -4];
                }
                const ratio = video.videoWidth && video.videoHeight
                    ? video.videoWidth / video.videoHeight
                    : 16 / 9;
                return [width, Math.max(120, (width - 20) / ratio + 10)];
            };

            video.addEventListener("loadedmetadata", () => {
                this.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            });

            video.addEventListener("error", () => {
                console.warn("DaSiWa AutoLong could not load the video preview.");
            });

            this.autolongVideoPreview = { container, video, download };
        });

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            const preview = message?.gifs?.[0];
            const elements = this.autolongVideoPreview;
            if (!preview || !elements) return;

            const params = {
                ...preview,
                timestamp: Date.now(),
            };
            elements.container.style.display = "block";
            elements.video.src = api.apiURL(
                `/view?${new URLSearchParams(params).toString()}`,
            );
            elements.download.href = elements.video.src;
            elements.download.download = preview.filename || "DaSiWa_AUTO_LONG.mp4";
            elements.video.load();
            elements.video.play().catch(() => {});
            this.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        });
    },
});
