import { create } from "zustand";
import { ProfileDataElementsType } from "../types";

const initialState = {
    profileData: {},
};

const useProfileDataStore = create((set, get) => ({
    ...initialState,
    updateProfileData: ((type: ProfileDataElementsType, data: any) => {

        set((state: any) => ({
            ...state,
            profileData: {
                ...state.profileData,
                [type]: data,   
            }                     
        }));

    }),


}));
export default useProfileDataStore;