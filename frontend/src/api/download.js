import axios from "axios";

export async function download_all_pdfs(folder) {
  const downloadUrl = `https://tarrifs-backend-app.azurewebsites.net/download-pdfs?folder=${encodeURIComponent(folder)}`;
  window.open(downloadUrl, "_blank");
}

