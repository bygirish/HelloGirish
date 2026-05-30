// app/posts/[id]/page.tsx
import { FC } from 'react';

interface PostProps {
  params: {
    id: string;
  };
}

const posts = [
  { id: '1', title: 'First Post', content: 'This is the content of the first post.' },
  { id: '2', title: 'Second Post', content: 'This is the content of the second post.' },
];

export async function generateStaticParams() {
  // Generate static parameters for all posts
  return posts.map(post => ({
    id: post.id,
  }));
}

const PostPage: FC<PostProps> = ({ params }) => {
  const post = posts.find(p => p.id === params.id);

  if (!post) {
    return <p>Post not found</p>;
  }

  return (
    <div>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </div>
  );
};

export default PostPage;
