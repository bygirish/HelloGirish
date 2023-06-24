import axios from "axios";
import { LoginAPI } from "../types";
import { ExtractFnReturnType, QueryConfig } from "@/lib/react-query";
import { UseQueryOptions, useQuery } from "@tanstack/react-query";


export const loginWithEmailAndPassword = (
    data: LoginAPI.requestPayload
  ): Promise<LoginAPI.Root> => {
    return axios.get('/user/v1/login').then(response => response.data);
};


export const useLoginWithEmailAndPassword = (payload: LoginAPI.requestPayload) => {
    useQuery({
        queryKey: ['loginWithEmail'],
        queryFn: loginWithEmailAndPassword(payload)
    })
}

// ExtractFnReturnType<typeof loginWithEmailAndPassword>

export const useGamesDashboardData = ({ config }: UseBhanzuPlayOptions = {}) => {
    return useQuery<ExtractFnReturnType<DashboardQueryFnType>>({
      ...config,
      queryKey: ['gamesDashboardData'],
      queryFn: () => getGamesDashboardData(),
    });
  };