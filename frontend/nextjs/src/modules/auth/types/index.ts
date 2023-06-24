import { BaseResponse } from "@/types/api";

export declare namespace LoginAPI {
    export interface requestPayload {
        emailId: string;
        password: string;
    }

    export interface Root extends BaseResponse {
        userId: string;

    }
}