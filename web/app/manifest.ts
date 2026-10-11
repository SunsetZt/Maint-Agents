import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "工业设备现场运维工作台",
    short_name: "现场运维",
    description: "企业级多模态工业设备智能运维与售后 Agent 平台现场工作区",
    id: "/field",
    start_url: "/field",
    scope: "/field/",
    display: "standalone",
    background_color: "#f5f7fa",
    theme_color: "#0b5cad",
    lang: "zh-CN",
    orientation: "any",
    icons: [
      {
        src: "/icon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "any",
      },
      {
        src: "/icon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "maskable",
      },
    ],
  };
}
