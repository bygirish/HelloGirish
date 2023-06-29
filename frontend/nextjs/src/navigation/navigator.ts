import { useRouter, usePathname } from "next/navigation";

const Navigator = () => {

    const router = useRouter();
    const {
         back,
         forward,
         refresh,
         push,
         replace,
         prefetch
    } = router;

    const currentPathName  = usePathname();

    return ({
        back,
        forward,
        refresh,
        push,
        replace,
        prefetch,
        currentPathName
    });
}


export default Navigator;
