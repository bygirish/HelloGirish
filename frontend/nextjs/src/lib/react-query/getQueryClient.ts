import { QueryClient } from "react-query";
import { cache } from "react";

const getQueryClient = cache(() => new QueryClient());
export default getQueryClient;

// https://blog.gogrow.dev/setting-up-react-query-in-your-next-js-13-app-e8edea0d20cc