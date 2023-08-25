// https://codevoweb.com/react-query-context-api-axios-interceptors-jwt-auth/#google_vignette

import { LoginAPI, SignupAPI, UserAPI } from "../types";
import APIClient from "@/lib/networking/apiClient";
import { ExtractFnReturnType } from "@/lib/react-query";
import { BaseResponse } from "@/types/api";
import { useMutation, useQuery } from "react-query";

export const signUpUserFn = async (user: SignupAPI.requestPayload) => {
  const response = await APIClient.post<SignupAPI.Response>(
    "auth/register",
    user
  );
  return response.data;
};

export const useSignUpUserMutation = ({
  onSuccessCb,
  onErrorCb
}: {
  onSuccessCb: (data: any) => void;
  onErrorCb: (data: any) => void;
}) => {
  return useMutation(
    (userData: SignupAPI.requestPayload) => signUpUserFn(userData),
    {
      onSuccess: onSuccessCb,
      // onSuccess(data) {
      //   toast.success(data?.message);
      //   navigate('/verifyemail');
      // },
      onError: onErrorCb,
    //   onError(error: any) {
    //     if (Array.isArray((error as any).response.data.error)) {
    //       (error as any).response.data.error.forEach((el: any) =>
    //         toast.error(el.message, {
    //           position: 'top-right',
    //         })
    //       );
    //     } else {
    //       toast.error((error as any).response.data.message, {
    //         position: 'top-right',
    //       });
    //     }
    //   },
    // }
    }
  );
};

export const loginUserFn = async (user: LoginAPI.requestPayload) => {
  const response = await APIClient.post<LoginAPI.Response>("auth/login", user);
  return response.data;
};

export const useLoginUserMutation = ({
  onSuccessCb,
  onErrorCb
}: {
  onSuccessCb: (data: any) => void;
  onErrorCb: (data: any) => void;
}) => {
  return useMutation(
    (userData: LoginAPI.requestPayload) => loginUserFn(userData),
    {
      onSuccess: onSuccessCb,
      // onSuccess: (data) => {
      //   query.refetch();
      //   toast.success('You successfully logged in');
      //   navigate(from);
      // },
      onError: onErrorCb,
      // onError: (error: any) => {
      //   if (Array.isArray((error as any).response.data.error)) {
      //     (error as any).response.data.error.forEach((el: any) =>
      //       toast.error(el.message, {
      //         position: 'top-right',
      //       })
      //     );
      //   } else {
      //     toast.error((error as any).response.data.message, {
      //       position: 'top-right',
      //     });
      //   }
      // },
    }
  );
};

export const verifyEmailFn = async (verificationCode: string) => {
  const response = await APIClient.get<BaseResponse>(
    `auth/verifyemail/${verificationCode}`
  );
  return response.data;
};

export const useVerifyEmailMutation = ({
  onSuccessCb,
  onErrorCb
}: {
  onSuccessCb: (data: any) => void;
  onErrorCb: (data: any) => void;
}) => {
  return useMutation(
    (verificationCode: string) => verifyEmailFn(verificationCode),
    {
      onSuccess: onSuccessCb,
      // onSuccess: (data) => {
      //   toast.success(data?.message);
      //   navigate('/login');
      // },
      onError: onErrorCb,
      // onError(error: any) {
      //   if (Array.isArray((error as any).data.error)) {
      //     (error as any).data.error.forEach((el: any) =>
      //       toast.error(el.message, {
      //         position: 'top-right',
      //       })
      //     );
      //   } else {
      //     toast.error((error as any).data.message, {
      //       position: 'top-right',
      //     });
      //   }
      // },
    }
  );
};


export const logoutUserFn = async () => {
  const response = await APIClient.get<BaseResponse>("auth/logout");
  return response.data;
};

export const useLogoutUserMutation = ({
  onSuccessCb,
  onErrorCb
}: {
  onSuccessCb: (data: any) => void;
  onErrorCb: (data: any) => void;
}) => {
  return useMutation(
    () => logoutUserFn(),
    {
      onSuccess: onSuccessCb,
      // onSuccess: (data) => {
      //   window.location.href = '/login';
      // },
      onError: onErrorCb,
      // onError: (error: any) => {
      //   if (Array.isArray(error.response.data.error)) {
      //     error.data.error.forEach((el: any) =>
      //       toast.error(el.message, {
      //         position: 'top-right',
      //       })
      //     );
      //   } else {
      //     toast.error(error.response.data.message, {
      //       position: 'top-right',
      //     });
      //   }
      // },
    }
  );
};

export const getMeFn = async () => {
  const response = await APIClient.get<UserAPI.Response>("users/me");
  return response.data;
};

export const useGetMeFn = ({
  onSuccessCb,
}: {
  onSuccessCb: (data: any) => void;
}) => {
  return useQuery<ExtractFnReturnType<typeof getMeFn>>({
    queryKey: ["authUser"],
    queryFn: () => getMeFn(),
    onSuccess: onSuccessCb,
    // cacheTime: 100,
  });
};

export const refreshAccessTokenFn = async () => {
  const response = await APIClient.get<LoginAPI.Response>("auth/refresh");
  return response.data;
};
