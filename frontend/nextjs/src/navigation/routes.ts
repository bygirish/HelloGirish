export const routePath = {
    home: {
        public: `/login`,
        private: '/dashboard',
    },
    auth: {
        login: '/login',
        register: '/register',
    },
    dashboard: {
        dashboard: '/dashboard',
    },
    profile: {
        viewProfile: (profileId: string) => `/profile/${profileId}`,
        editProfile: (profileId: string) => `/profile/${profileId}/edit`,
    },
    blog: {
        blogListing: `blog`,
        blogView: (id: string) => `/blog/${id}`,
        blogEdit:  (id: string) => `/blog/${id}/edit`,
        blogPreview: (id: string) => `/blog/${id}/preview`,
    }
}