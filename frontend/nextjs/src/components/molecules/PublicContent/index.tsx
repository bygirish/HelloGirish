export const PublicContent = ({ children }: { children: React.ReactNode }) => {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: 'column'
      }}
    >
      {children}
    </div>
  );
};
