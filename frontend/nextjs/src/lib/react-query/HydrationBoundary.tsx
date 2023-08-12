

// import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
// import getQueryClient from "./getQueryClient";

// const HydrationBoundaryComponent = async ({
//   children,
//   prefetchQueriesData
// }: {
//   children: React.ReactNode;
//   prefetchQueriesData: {
//     queryKey: string;
//     queryFunction: Function;
//   }
// }) => {
//   const queryClient = getQueryClient();


//   await
//   await queryClient.prefetchQuery({ querykey: ["posts"], queryFn: getPosts });
//   // for infinite queries with useInfiniteQuery use
//   // await queryClient.prefetchInfiniteQuery({queryKey:['posts'], queryFn:getPosts,...})
//   const dehydratedState = dehydrate(queryClient);

//   return (
//     <HydrationBoundary state={dehydratedState}>{children}</HydrationBoundary>
//   );
// };

// export default HydrationBoundaryComponent;

