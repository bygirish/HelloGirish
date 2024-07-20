import { ProfileDataElements } from "./constants";



export type ProfileDataElementsKeysType = keyof typeof ProfileDataElements;
export type ProfileDataElementsType = typeof ProfileDataElements[ProfileDataElementsKeysType];



