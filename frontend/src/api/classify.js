import axios from "axios";

export async function classify(description) {
  const response = await axios.post('https://tarrifs-backend-app.azurewebsites.net/classify', { description });
  return response.data;
}
