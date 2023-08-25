import { BaseResponse } from "@/types/api";

export declare namespace SignupAPI {
  export interface requestPayload {
    name: string;
    emailId: string;
    password: string;
  }

  export interface Response extends BaseResponse {

  }
}

export declare namespace LoginAPI {
  export interface requestPayload {
    emailId: string;
    password: string;
  }

  export interface Response extends BaseResponse {
    access_token: string;
  }
}

export declare namespace UserAPI {
  export interface requestPayload {}

  export interface Response extends BaseResponse {
    data: {
      user: IUser;
    };
  }
}

export interface IUser {
  name: string;
  email: string;
  role: string;
  _id: string;
  id: string;
  createdAt: string;
  updatedAt: string;
  __v: number;
}

// export interface ILoginResponse {
//   status: string;
//   access_token: string;
// }

// export interface IUserResponse {
//   status: string;
//   data: {
//     user: IUser;
//   };
// }
