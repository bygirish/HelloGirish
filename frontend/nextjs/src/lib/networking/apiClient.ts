import { refreshAccessTokenFn } from "@/modules/auth/api";
import { LoginAPI } from "@/modules/auth/types";

// https://codevoweb.com/react-query-context-api-axios-interceptors-jwt-auth/#google_vignette
import Axios, { InternalAxiosRequestConfig } from "axios";
const BASE_URL = "http://localhost:8000/api/";

export const APIClient = Axios.create({
  baseURL: BASE_URL,
  withCredentials: true,
});

function authRequestInterceptor(config: InternalAxiosRequestConfig) {
  // const token = tokenStorage.getToken();
  // if (token) {
  //   config.headers.authorization = `${token}`;
  // }
  if (config.headers) config.headers.Accept = "application/json";
  return config;
}

APIClient.interceptors.request.use(authRequestInterceptor);

APIClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    const errMessage = error.response.data.message as string;
    if (errMessage.includes("not logged in") && !originalRequest._retry) {
      originalRequest._retry = true;
      await refreshAccessTokenFn();
      return APIClient(originalRequest);
    }
    return Promise.reject(error);
  }
);

export default APIClient;